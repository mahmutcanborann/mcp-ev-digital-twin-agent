from agent.discovery_mcp_agent import DiscoveryMCPAgent

agent = DiscoveryMCPAgent()

print(agent.ask("Compare B0005 and B0047"))
print(agent.ask("Which battery is the riskiest?"))
print(agent.ask("How healthy is B0005?"))