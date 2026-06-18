from agent.ev_agent import EVDigitalTwinAgent

agent = EVDigitalTwinAgent()

print(agent.ask("How healthy is my battery?"))
print(agent.ask("What is the estimated range?"))
print(agent.ask("What maintenance actions do you recommend?"))
print(agent.ask("How is my charging behavior?"))
print(agent.ask("How is my driving style?"))
print(agent.ask("What is the overall score?"))
print(agent.ask("Compare B0005 and B0047"))
print(agent.ask("Which battery is the riskiest?"))
print(agent.ask("Which battery is the healthiest?"))