from src.fleet_anomaly_detector import FleetAnomalyDetector

detector = FleetAnomalyDetector()

print(detector.get_most_anomalous_battery())

print(detector.detect_anomalies())