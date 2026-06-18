from src.battery_twin import BatteryDigitalTwin

twin = BatteryDigitalTwin()

print("SOH:", twin.predict_soh("B0005", 150, 24))
print("RUL:", twin.estimate_rul("B0005", 120, 80))
print("Range:", twin.estimate_range(80, 500))
print("Future:", twin.simulate_future("B0005", 120, 125))
print("4°C SOH:", twin.predict_soh("B0005", 150, 4))
print("24°C SOH:", twin.predict_soh("B0005", 150, 24))
print("44°C SOH:", twin.predict_soh("B0005", 150, 44))

print("Summary:", twin.generate_summary("B0005", 150, 24, 500))

for temp in [4, 24, 44]:
    print(f"B0038 - {temp}°C:", twin.predict_soh("B0038", 40, temp))

for temp in [4, 22, 24]:
    print(f"B0042 - {temp}°C:", twin.predict_soh("B0042", 40, temp))

    scenarios = twin.simulate_what_if_scenarios(
    battery_id="B0005",
    start_cycle=80,
    end_cycle=120,
    ambient_temperature=24,
    nominal_range_km=500
)

for name, values in scenarios.items():
    print(name, values[:3])