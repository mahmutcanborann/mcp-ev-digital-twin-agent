import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.battery_knowledge_base import BatteryKnowledgeBase

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
LLM_SELECTION_ATTEMPTS = 2

# Keyword fallback used when the LLM is unreachable or returns an unusable answer.
# Order matters: the first matching rule wins (mirrors the rules in the LLM prompt).
FALLBACK_RULES = [
    (r"\bcompare\b", "compare_batteries"),
    (r"\bhealthiest\b|\bbest battery\b", "get_healthiest_battery"),
    (r"\briskiest\b|\bworst battery\b|\bmost risky\b", "get_riskiest_battery"),
    (r"measurement-based|\bmeasured\b|\bvalidated\b|out-of-sample|how accurate", "estimate_soh_for_battery_cycle"),
    (r"\brul\b|remaining useful life", "estimate_rul"),
    (r"\bwhat[- ]if\b|\bscenario", "simulate_what_if_scenarios"),
    (r"\bcharg", "get_charging_summary"),
    (r"\bdriv", "get_driving_score"),
    (r"\bdrift\b|\bdeviation\b|\boutlier", "check_soh_deviation"),
    (r"system health|system status|\bwarnings?\b", "check_system_health"),
]
DEFAULT_TOOL = "generate_summary"


class LLMUnavailableError(RuntimeError):
    """Ollama could not be reached or returned an unusable HTTP response."""


class DiscoveryMCPAgent:
    def __init__(self, ollama_url: str | None = None, ollama_model: str | None = None):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(BASE_DIR / "mcp_servers" / "battery_soh_server.py")],
            cwd=str(BASE_DIR)
        )
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.last_selected_tool = None
        self.last_selection_method = None   # "llm", "rule_based_fallback" or "multi_step"
        self.knowledge_base = BatteryKnowledgeBase()
        fleet_path = BASE_DIR / "models" / "fleet_latest_status.csv"
        self.fleet_df = pd.read_csv(fleet_path)

    # ------------------------------------------------------------------ LLM I/O
    def _call_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMUnavailableError(
                f"LLM not reachable at {self.ollama_url} (model {self.ollama_model}): {exc}"
            ) from exc

    # ------------------------------------------------------------ tool selection
    @staticmethod
    def parse_tool_selection(raw: str, valid_tools: set) -> dict | None:
        """Extract {"tool_name", "arguments"} from an LLM answer; None if unusable."""
        if not isinstance(raw, str):
            return None
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end])
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("tool_name") not in valid_tools:
            return None
        arguments = data.get("arguments")
        return {"tool_name": data["tool_name"], "arguments": arguments if isinstance(arguments, dict) else {}}

    @staticmethod
    def rule_based_tool(question: str, valid_tools: set) -> str:
        q = question.lower()
        for pattern, tool in FALLBACK_RULES:
            if tool in valid_tools and re.search(pattern, q):
                return tool
        return DEFAULT_TOOL if DEFAULT_TOOL in valid_tools else sorted(valid_tools)[0]

    def select_tool(self, question: str, tools_text: str, valid_tools: set) -> tuple[str, dict, str]:
        """Ask the LLM (with one retry); fall back to keyword rules if it fails."""
        for _ in range(LLM_SELECTION_ATTEMPTS):
            try:
                raw = self.ask_gemma_for_tool(question, tools_text)
            except LLMUnavailableError:
                break
            selection = self.parse_tool_selection(raw, valid_tools)
            if selection:
                return selection["tool_name"], selection["arguments"], "llm"
        return self.rule_based_tool(question, valid_tools), {}, "rule_based_fallback"

    def ask_gemma_for_tool(self, question: str, tools_text: str) -> str:
        prompt = f"""
You are an MCP tool selection agent for an EV Digital Twin system.

User question:
{question}

Available MCP tools:
{tools_text}

Select the best tool for the user question.

Return ONLY valid JSON in this format:
{{
  "tool_name": "tool_name_here",
  "arguments": {{}}
}}

Rules:
- Do not explain.
- Do not use markdown.
- Use only one of the available tool names.
- If the question asks to compare batteries, use compare_batteries.
- If the question asks for healthiest battery, use get_healthiest_battery.
- If the question asks for riskiest or worst battery, use get_riskiest_battery.
- If the question asks for a measurement-based, measured, validated or out-of-sample SOH estimate, or how accurate the SOH estimate is for a battery or cycle, use estimate_soh_for_battery_cycle.
- If the question asks about SOH or battery health, use generate_summary.
- If the question asks about RUL or remaining useful life, use estimate_rul.
- If the question asks about charging, use get_charging_summary.
- If the question asks about driving, use get_driving_score.
- If the question asks about system health or warnings, use check_system_health.
- If the question asks about drift, deviation or outliers compared to the fleet, use check_soh_deviation.
- If the question asks about what-if scenarios, use simulate_what_if_scenarios.
- If the user asks "how healthy is battery" or asks for the health of a specific battery, use generate_summary instead of check_system_health.
- If the user asks why a specific battery is risky, degraded, or unhealthy, use generate_summary for that specific battery.
- If the user asks which battery is riskiest or worst in the fleet, use get_riskiest_battery.
"""

        raw = self._call_ollama(prompt).strip()
        print("Gemma raw tool selection:")
        print(raw)
        return raw

    async def ask_async(
        self,
        question: str,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                if self.is_multi_step_question(question):
                    results = {}

                    summary_result = await session.call_tool(
                        "generate_summary",
                        arguments={
                            "battery_id": battery_id,
                            "cycle": cycle,
                            "ambient_temperature": ambient_temperature,
                            "nominal_range_km": nominal_range_km
                        }
                    )

                    rul_result = await session.call_tool(
                        "estimate_rul",
                        arguments={
                            "battery_id": battery_id,
                            "current_cycle": cycle,
                            "threshold": 80,
                            "ambient_temperature": ambient_temperature
                        }
                    )

                    health_result = await session.call_tool(
                        "check_system_health",
                        arguments={
                            "battery_id": battery_id,
                            "cycle": cycle,
                            "ambient_temperature": ambient_temperature,
                            "nominal_range_km": nominal_range_km
                        }
                    )

                    results["summary"] = summary_result.content[0].text
                    results["rul"] = rul_result.content[0].text
                    results["system_health"] = health_result.content[0].text

                    self.last_selected_tool = "multi_step_health_assessment"
                    self.last_selection_method = "multi_step"

                    return self.explain_or_raw(
                        question=question,
                        tool_name="multi_step_health_assessment",
                        tool_result=json.dumps(results, indent=2)
                    )
                tools_text = "\n".join(
                    [
                        f"- {tool.name}: {tool.inputSchema}"
                        for tool in tools_result.tools
                    ]
                )

                valid_tools = {tool.name for tool in tools_result.tools}
                tool_name, arguments, method = self.select_tool(
                    question=question,
                    tools_text=tools_text,
                    valid_tools=valid_tools
                )
                self.last_selected_tool = tool_name
                self.last_selection_method = method

                arguments = self.fill_missing_arguments(
                    tool_name=tool_name,
                    arguments=arguments,
                    question=question,
                    battery_id=battery_id,
                    cycle=cycle,
                    ambient_temperature=ambient_temperature,
                    nominal_range_km=nominal_range_km
                )

                schema = next(t.inputSchema for t in tools_result.tools if t.name == tool_name)
                arguments = self.filter_arguments(arguments, schema)

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments
                )

                tool_result = result.content[0].text if result.content else ""
                if getattr(result, "isError", False):
                    return f"⚠️ The tool '{tool_name}' returned an error: {tool_result}"

                explanation = self.explain_or_raw(
                    question=question,
                    tool_name=tool_name,
                    tool_result=tool_result
                )

                # Numbers are appended from the tool result, not written by the LLM,
                # so they are always exactly what the model returned.
                return explanation + self.format_validation(tool_result)

    @staticmethod
    def filter_arguments(arguments: dict, schema: dict) -> dict:
        """Drop arguments the tool does not accept (LLMs sometimes invent parameter names)."""
        allowed = set((schema or {}).get("properties", {}))
        return {k: v for k, v in arguments.items() if k in allowed} if allowed else {}

    @staticmethod
    def format_validation(tool_result: str) -> str:
        """Validation block for out-of-sample SOH estimates (empty for other tools)."""
        try:
            data = json.loads(tool_result)
        except (TypeError, ValueError):
            return ""
        if not isinstance(data, dict) or "measured_soh" not in data:
            return ""

        lines = [
            "",
            "",
            "Validation (this battery was NOT used to train the model):",
            f"Estimated SOH: {data['estimated_soh']}%",
            f"Measured SOH: {data['measured_soh']}%",
            f"Estimation Error: {data['absolute_error']} SOH points "
            f"(typical error on unseen batteries: {data.get('typical_error_soh_points', 'n/a')})",
        ]
        if data.get("note"):
            lines.append(f"Note: {data['note']}")
        return "\n".join(lines)

    def explain_or_raw(self, question: str, tool_name: str, tool_result: str) -> str:
        """LLM explanation, or the raw tool result with a notice if the LLM is unavailable."""
        try:
            return self.explain_result_with_gemma(question, tool_name, tool_result)
        except LLMUnavailableError as exc:
            return (
                f"⚠️ The language model is not available, so the tool result is shown without "
                f"an explanation.\n({exc})\n\nTool result ({tool_name}):\n{tool_result}"
            )

    def fill_missing_arguments(
        self,
        tool_name: str,
        arguments: dict,
        question: str,
        battery_id: str,
        cycle: int,
        ambient_temperature: int,
        nominal_range_km: float
    ) -> dict:
        def set_if_missing(key, value):
            if key not in arguments or arguments[key] is None:
                arguments[key] = value
        detected_battery_id = self.extract_battery_id(question, battery_id)
        fleet_context = self.get_fleet_battery_context(detected_battery_id)
        battery_id = detected_battery_id
        cycle = fleet_context.get("cycle", cycle)
        ambient_temperature = fleet_context.get("ambient_temperature", ambient_temperature)
        if tool_name in ["generate_summary", "predict_soh"]:
            set_if_missing("battery_id", battery_id)
            set_if_missing("cycle", cycle)
            set_if_missing("ambient_temperature", ambient_temperature)
            if tool_name == "generate_summary":
                set_if_missing("nominal_range_km", nominal_range_km)
        if tool_name == "estimate_rul":
            arguments.setdefault("battery_id", battery_id)
            arguments.setdefault("current_cycle", cycle)
            arguments.setdefault("threshold", 80)
            arguments.setdefault("ambient_temperature", ambient_temperature)

        if tool_name == "compare_batteries":
            arguments.setdefault("question", question)
            arguments.setdefault("cycle", cycle)
            arguments.setdefault("ambient_temperature", ambient_temperature)
            arguments.setdefault("nominal_range_km", nominal_range_km)

        if tool_name == "check_system_health":
            arguments.setdefault("battery_id", battery_id)
            arguments.setdefault("cycle", cycle)
            arguments.setdefault("ambient_temperature", ambient_temperature)
            arguments.setdefault("nominal_range_km", nominal_range_km)

        if tool_name == "estimate_soh_for_battery_cycle":
            arguments.setdefault("battery_id", battery_id)
            cycle_match = re.search(r"cycle\s*(\d+)", question.lower())
            arguments.setdefault(
                "cycle",
                int(cycle_match.group(1)) if cycle_match else cycle
            )

        if tool_name == "check_soh_deviation":
            arguments.setdefault("current_soh", fleet_context.get("fleet_soh", 75.0))

        if tool_name == "simulate_what_if_scenarios":
            arguments.setdefault("battery_id", battery_id)
            arguments.setdefault("start_cycle", cycle)
            arguments.setdefault("end_cycle", cycle + 50)
            arguments.setdefault("ambient_temperature", ambient_temperature)
            arguments.setdefault("nominal_range_km", nominal_range_km)
        print("Final tool arguments:")
        print(arguments)
        return arguments
    def explain_result_with_gemma(
        self,
        question: str,
        tool_name: str,
        tool_result: str
    ) -> str:
        context = self.knowledge_base.retrieve_context(question)
        prompt = f"""
You are an EV Digital Twin AI assistant.

User question:
{question}

Selected tool:
{tool_name}

Tool result:
{tool_result}
Battery knowledge context:
{context}
Explain the result clearly and professionally.

Rules:
- Do not mention MCP.
- Do not mention tool names.
- Do not invent values.
- Do not mention measured SOH, absolute error or typical error; a validation section is appended automatically.
- Copy Health Status exactly from the tool result. If the tool result has no health status, write "Not available".
- Give charging, temperature or maintenance advice from the knowledge context ONLY if the tool result shows SOH below 80% or an explicit risk or warning. Otherwise recommend only continued monitoring.
-Never assume or infer battery degradation mechanisms unless they are explicitly present in the tool result.
If information is missing, simply state that it is not available.
- Use clean spacing.
- Focus on EV battery engineering implications.
- Use recommendations that are consistent with the actual battery data.
- Do not provide temperature-related recommendations unless supported by the tool result.
- Do not recommend avoiding heat exposure for batteries operating in cold conditions.
- Base recommendations only on the information available in the tool result and battery knowledge context.
- Do not claim that temperature is causing degradation unless the tool result explicitly indicates a temperature effect.
- Distinguish between correlation and causation.
Use the Battery Health Report format ONLY if the user asks for:
- health of a specific battery
- assessment of a specific battery
- complete assessment
- full report
- system status

DO NOT use Battery Health Report format for comparison questions.
Use EXACTLY this format:

Battery Health Report

Battery ID: ...
SOH: ...%
Estimated Range: ... km
Health Status: ...

System Assessment:
...

Recommendation:
...

If the question asks for battery comparison:
- clearly compare the batteries
- state which battery is healthier

If the question asks for the riskiest battery:
- explain why it is considered risky
"""

        return self._call_ollama(prompt)

    def ask(
        self,
        question: str,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        try:
            return asyncio.run(
                self.ask_async(
                    question=question,
                    battery_id=battery_id,
                    cycle=cycle,
                    ambient_temperature=ambient_temperature,
                    nominal_range_km=nominal_range_km
                )
            )
        except Exception as exc:  # MCP server crash, tool error, ... -> readable message
            self.last_selected_tool = None
            self.last_selection_method = None
            detail = exc
            while isinstance(detail, BaseExceptionGroup) and detail.exceptions:
                detail = detail.exceptions[0]
            message = f"⚠️ The agent could not complete this request: {type(detail).__name__}: {detail}"
            if "connection closed" in str(detail).lower():
                message += (
                    "\nThe MCP server stopped during start-up. Run "
                    "`python mcp_servers/battery_soh_server.py` to see the error "
                    "(a common cause is mcp 2.x; install requirements.txt, which pins mcp<2)."
                )
            return message
    def is_multi_step_question(self, question: str) -> bool:
        q = question.lower()

        multi_step_keywords = [
            "complete assessment",
            "full assessment",
            "full report",
            "complete report",
            "overall assessment",
            "detailed report",
            "comprehensive",
            "overall health"
        ]

        return any(keyword in q for keyword in multi_step_keywords)

    def extract_battery_id(self, question: str, default_battery_id: str) -> str:
        match = re.search(r"B\d{4}", question.upper())

        if match:
            return match.group(0)

        return default_battery_id


    def get_fleet_battery_context(self, battery_id: str) -> dict:
        row = self.fleet_df[self.fleet_df["battery_id"] == battery_id]

        if row.empty:
            return {}

        row = row.iloc[0]

        return {
            "battery_id": battery_id,
            "cycle": int(row["cycle"]),
            "ambient_temperature": int(row["ambient_temperature"]),
            "fleet_soh": round(float(row["SOH"]), 2)
        }