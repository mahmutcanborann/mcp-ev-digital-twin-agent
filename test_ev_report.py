from src.ev_twin_report import EVTwinReport

report_engine = EVTwinReport()

report = report_engine.generate_report(
    battery_id="B0005",
    cycle=150,
    ambient_temperature=24,
    nominal_range_km=500
)

print(report)