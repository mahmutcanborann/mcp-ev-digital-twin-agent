from pathlib import Path


class BatteryKnowledgeBase:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[1]
        self.kb_path = base_dir / "knowledge_base" / "battery_degradation.md"
        self.knowledge_text = self.kb_path.read_text(encoding="utf-8")

    def retrieve_context(self, question: str) -> str:
        q = question.lower()

        keywords = [
            "degraded",
            "degradation",
            "risky",
            "risk",
            "soh",
            "health",
            "maintenance",
            "temperature",
            "fast charging",
            "replacement",
            "anomaly"
        ]

        if any(keyword in q for keyword in keywords):
            return self.knowledge_text

        return ""