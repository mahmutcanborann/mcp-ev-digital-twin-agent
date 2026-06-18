from src.drift_detector import DataDriftDetector

detector = DataDriftDetector()

print(detector.detect_soh_drift(77.2))
print(detector.detect_soh_drift(94.8))
print(detector.detect_soh_drift(29.0))