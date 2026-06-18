from agent.llm_ev_agent import LLMEVDigitalTwinAgent

agent = LLMEVDigitalTwinAgent()

answer = agent.ask(
    question="Why is this battery risky?",
    battery_id="B0005",
    cycle=150,
    ambient_temperature=24
)

print(answer)