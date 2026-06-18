from agent.mcp_tool_agent import MCPToolAgent

agent = MCPToolAgent()

print(agent.ask("Compare B0005 and B0047"))
print()
print(agent.ask("Which battery is the riskiest?"))
print()
print(agent.ask("Which battery is the healthiest?"))
print(agent.ask("What is the battery health?"))
print()
print(agent.ask("What is the remaining useful life?"))
print()
print(agent.ask("How is the driving score?"))
print(agent.ask("How is the charging behavior?"))
print()
print(agent.ask("What is the system monitoring status?"))
print()
print(agent.ask("Is there any data drift?"))