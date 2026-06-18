import re

from agent.ev_agent import EVDigitalTwinAgent
import requests

class ToolCallingEVAgent:
    def __init__(self):
        self.base_agent = EVDigitalTwinAgent()

        self.tools = {
            "compare_batteries": self.compare_batteries,
            "get_riskiest_battery": self.get_riskiest_battery,
            "get_healthiest_battery": self.get_healthiest_battery
        }
    def explain_with_gemma(self, question: str, tool_name: str, tool_result: str) -> str:
        prompt = f"""
You are an EV Digital Twin AI assistant.

The user asked:
{question}

The selected tool was:
{tool_name}

The tool returned this result:
{tool_result}

Explain the result clearly and professionally.
Do not mention internal tool names.
Do not invent values.
Use clean spacing and concise formatting.
Answer in 3-5 sentences.
Always include one final sentence starting with "Maintenance implication:".
This sentence should explain what the result means for monitoring, inspection, or replacement priority.
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"]

    def route_tool(self, question: str) -> str:
        q = question.lower()

        if "compare" in q:
            return "compare_batteries"

        if "riskiest" in q or "worst" in q or "most risky" in q:
            return "get_riskiest_battery"

        if "healthiest" in q or "best" in q:
            return "get_healthiest_battery"

        return "general_answer"

    def ask(
        self,
        question: str,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        tool_name = self.route_tool(question)

        if tool_name == "general_answer":
            return self.base_agent.ask(
                question=question,
                battery_id=battery_id,
                cycle=cycle,
                ambient_temperature=ambient_temperature,
                nominal_range_km=nominal_range_km
            )

        tool = self.tools[tool_name]

        return tool(
            question=question,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

    def compare_batteries(
        self,
        question: str,
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        battery_ids = re.findall(r"B\d{4}", question.upper())

        if len(battery_ids) < 2:
            return (
                "Tool selected: compare_batteries. "
                "Please provide two battery IDs, for example: Compare B0005 and B0047."
            )

        tool_result = self.base_agent.compare_batteries(
            question=question,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

        return self.explain_with_gemma(
            question=question,
            tool_name="compare_batteries",
            tool_result=tool_result
        )

    def get_riskiest_battery(
        self,
        question: str = "",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        tool_result = self.base_agent.get_riskiest_battery()

        return self.explain_with_gemma(
            question=question,
            tool_name="get_riskiest_battery",
            tool_result=tool_result
        )

    def get_healthiest_battery(
        self,
        question: str = "",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        tool_result = self.base_agent.get_healthiest_battery()

        return self.explain_with_gemma(
            question=question,
            tool_name="get_healthiest_battery",
            tool_result=tool_result
        )