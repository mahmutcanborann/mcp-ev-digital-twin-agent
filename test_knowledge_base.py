from src.battery_knowledge_base import BatteryKnowledgeBase

kb = BatteryKnowledgeBase()

context = kb.retrieve_context("Why is B0049 risky?")

print(context)