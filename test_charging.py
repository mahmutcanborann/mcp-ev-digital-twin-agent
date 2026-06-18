from src.charging_analyzer import ChargingAnalyzer

analyzer = ChargingAnalyzer()

print(analyzer.get_summary())
print(analyzer.risk_distribution())