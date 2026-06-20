import asyncio
import json
import os
import requests
import re
import pandas as pd
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.battery_knowledge_base import BatteryKnowledgeBase

class DiscoveryMCPAgent:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command="python",
            args=["mcp_servers/battery_soh_server.py"]
        )
        self.last_selected_tool = None
        self.knowledge_base = BatteryKnowledgeBase()
        base_dir = Path(__file__).resolve().parents[1]
        fleet_path = base_dir / "models" / "fleet_latest_status.csv"
        self.fleet_df = pd.read_csv(fleet_path)
    def ask_gemma_for_tool(self, question: str, tools_text: str) -> dict:
        ollama_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate"
        )

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
- If the question asks about SOH or battery health, use generate_summary.
- If the question asks about RUL or remaining useful life, use estimate_rul.
- If the question asks about charging, use get_charging_summary.
- If the question asks about driving, use get_driving_score.
- If the question asks about system health or warnings, use check_system_health.
- If the question asks about drift, use detect_soh_drift.
- If the user asks "how healthy is battery" or asks for the health of a specific battery, use generate_summary instead of check_system_health.
- If the user asks why a specific battery is risky, degraded, or unhealthy, use generate_summary for that specific battery.
- If the user asks which battery is riskiest or worst in the fleet, use get_riskiest_battery.
"""

        response = requests.post(
            ollama_url,
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        raw = response.json()["response"].strip()

        print("Gemma raw tool selection:")
        print(raw)

        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in Gemma response: {raw}")

        json_text = raw[start:end]

        return json.loads(json_text)

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

                    return self.explain_result_with_gemma(
                        question=question,
                        tool_name="multi_step_health_assessment",
                        tool_result=str(results)
                    )
                tools_text = "\n".join(
                    [
                        f"- {tool.name}: {tool.inputSchema}"
                        for tool in tools_result.tools
                    ]
                )

                selection = self.ask_gemma_for_tool(
                    question=question,
                    tools_text=tools_text
                )

                tool_name = selection["tool_name"]
                self.last_selected_tool = tool_name

                arguments = selection.get("arguments", {})

                arguments = self.fill_missing_arguments(
                    tool_name=tool_name,
                    arguments=arguments,
                    question=question,
                    battery_id=battery_id,
                    cycle=cycle,
                    ambient_temperature=ambient_temperature,
                    nominal_range_km=nominal_range_km
                )

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments
                )

                tool_result = result.content[0].text
               
                return self.explain_result_with_gemma(
                    question=question,
                    tool_name=tool_name,
                    tool_result=tool_result
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

        if tool_name == "detect_soh_drift":
            arguments.setdefault("current_soh", 75.0)

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

        ollama_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate"
        )
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

        response = requests.post(
            ollama_url,
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]

    def ask(
        self,
        question: str,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        return asyncio.run(
            self.ask_async(
                question=question,
                battery_id=battery_id,
                cycle=cycle,
                ambient_temperature=ambient_temperature,
                nominal_range_km=nominal_range_km
            )
        )
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