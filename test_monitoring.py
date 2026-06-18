from src.monitoring_engine import MonitoringEngine

engine = MonitoringEngine()

print(
    engine.check_system_health(
        battery_id="B0005",
        cycle=150,
        ambient_temperature=24
    )
)
print(engine.monitor_fleet())