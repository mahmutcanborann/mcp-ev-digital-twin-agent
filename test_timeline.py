from src.battery_twin import BatteryDigitalTwin

twin = BatteryDigitalTwin()

timeline = twin.generate_timeline(
    battery_id="B0005",
    start_cycle=0,
    end_cycle=200,
    step=50,
    ambient_temperature=24,
    nominal_range_km=500
)

for row in timeline:
    print(row)