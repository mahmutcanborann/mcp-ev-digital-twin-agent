from src.predictive_maintenance import (
    PredictiveMaintenanceEngine
)

engine = PredictiveMaintenanceEngine()

result = engine.predict_actions(
    battery_id="B0005",
    cycle=150,
    ambient_temperature=24
)

print(result)