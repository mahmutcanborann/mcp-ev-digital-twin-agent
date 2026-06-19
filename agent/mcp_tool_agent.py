import os
import asyncio
import re
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPToolAgent:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command="python",
            args=["mcp_servers/battery_soh_server.py"]
        )

    def route_tool(self, question: str):
        q = question.lower()

        if "compare" in q:
            return "compare_batteries"

        if "riskiest" in q or "worst" in q or "most risky" in q:
            return "get_riskiest_battery"

        if "healthiest" in q or "best" in q:
            return "get_healthiest_battery"

        if "soh" in q or "health" in q:
            return "generate_summary"

        if "rul" in q or "remaining" in q or "80%" in q:
            return "estimate_rul"

        if "driving" in q or "drive" in q or "driver" in q:
            return "get_driving_score"

        if "charging" in q or "charge" in q:
            return "get_charging_summary"

        if "system" in q or "monitoring" in q or "warning" in q:
            return "check_system_health"

        if "drift" in q:
            return "detect_soh_drift"

        return "generate_summary"

    async def ask_async(
        self,
        question: str,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ):
        tool_name = self.route_tool(question)

        if tool_name is None:
            return "No MCP tool was selected for this question."

        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                if tool_name == "compare_batteries":
                    result = await session.call_tool(
                        "compare_batteries",
                        arguments={
                            "question": question,
                            "cycle": cycle,
                            "ambient_temperature": ambient_temperature,
                            "nominal_range_km": nominal_range_km
                        }
                    )

                elif tool_name == "get_riskiest_battery":
                    result = await session.call_tool(
                        "get_riskiest_battery",
                        arguments={
                            "battery_id": battery_id
                        }
                    )

                elif tool_name == "get_healthiest_battery":
                    result = await session.call_tool(
                        "get_healthiest_battery",
                        arguments={
                            "battery_id": battery_id
                        }
                    )
                elif tool_name == "predict_soh":
                    result = await session.call_tool(
                        "predict_soh",
                        arguments={
                            "battery_id": battery_id,
                            "cycle": cycle,
                            "ambient_temperature": ambient_temperature
                        }
                    )

                elif tool_name == "estimate_rul":
                    result = await session.call_tool(
                        "estimate_rul",
                        arguments={
                            "battery_id": battery_id,
                            "current_cycle": cycle,
                            "threshold": 80,
                            "ambient_temperature": ambient_temperature
                        }
                    )

                elif tool_name == "generate_summary":
                    result = await session.call_tool(
                        "generate_summary",
                        arguments={
                            "battery_id": battery_id,
                            "cycle": cycle,
                            "ambient_temperature": ambient_temperature,
                            "nominal_range_km": nominal_range_km
                        }
                    )

                elif tool_name == "get_driving_score":
                    result = await session.call_tool(
                        "get_driving_score",
                        arguments={}
                    )

                elif tool_name == "get_charging_summary":
                    result = await session.call_tool(
                        "get_charging_summary",
                        arguments={}
                    )

                elif tool_name == "check_system_health":
                    result = await session.call_tool(
                        "check_system_health",
                        arguments={
                            "battery_id": battery_id,
                            "cycle": cycle,
                            "ambient_temperature": ambient_temperature,
                            "nominal_range_km": nominal_range_km
                        }
                    )

                elif tool_name == "detect_soh_drift":
                    result = await session.call_tool(
                        "detect_soh_drift",
                        arguments={
                            "current_soh": 72.56
                        }
                    )

                else:
                    return "Tool routing failed."

                tool_result = result.content[0].text

            return self.explain_with_gemma(
                question=question,
                tool_name=tool_name,
                tool_result=tool_result
            )

    def ask(
        self,
        question: str,
        cycle: int = 150,
        battery_id: str = "B0005",
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ):
        return asyncio.run(
            self.ask_async(
                question=question,
                cycle=cycle,
                battery_id=battery_id,
                ambient_temperature=ambient_temperature,
                nominal_range_km=nominal_range_km
            )
        )
    def explain_with_gemma(
        self,
        question: str,
        tool_name: str,
        tool_result: str
    ) -> str:
        prompt = f"""
You are an EV Digital Twin AI assistant.

The user asked:
{question}

The MCP tool selected was:
{tool_name}

The MCP tool returned:
{tool_result}

Explain the result clearly and professionally.
Do not mention internal tool names or MCP details.
Do not invent values.
Use clean spacing and concise formatting.
Answer in 3-5 sentences.
Always include one final sentence starting with "Maintenance implication:".
This sentence should explain what the result means for monitoring, inspection, or replacement priority.
"""

        ollama_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate"
        )

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