from src.driving_analyzer import DrivingAnalyzer

analyzer = DrivingAnalyzer()

print(analyzer.get_energy_summary())
print(analyzer.get_driving_score())
print(analyzer.get_driving_style())