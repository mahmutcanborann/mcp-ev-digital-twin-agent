import asyncio
import json
import os
import requests

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class DiscoveryMCPAgent:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command="python",
            args=["mcp_servers/battery_soh_server.py"]
        )

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
        if tool_name in ["generate_summary", "predict_soh"]:
            arguments.setdefault("battery_id", battery_id)
            arguments.setdefault("cycle", cycle)
            arguments.setdefault("ambient_temperature", ambient_temperature)

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

        prompt = f"""
You are an EV Digital Twin AI assistant.

User question:
{question}

Selected tool:
{tool_name}

Tool result:
{tool_result}

Explain the result clearly and professionally.

Rules:
- Do not mention MCP.
- Do not mention tool names.
- Do not invent values.
- Use 3-5 sentences.
- Focus on EV battery engineering implications.
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