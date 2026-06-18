from agent.tool_calling_ev_agent import ToolCallingEVAgent

agent = ToolCallingEVAgent()

print(agent.ask("Compare B0005 and B0047"))
print()
print(agent.ask("Which battery is the riskiest?"))
print()
print(agent.ask("Which battery is the healthiest?"))
print()
print(agent.ask("How healthy is my battery?"))