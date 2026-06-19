import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest


class FleetAnomalyDetector:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[1]
        fleet_path = base_dir / "models" / "fleet_latest_status.csv"
        self.fleet_df = pd.read_csv(fleet_path)

    def detect_anomalies(self) -> pd.DataFrame:
        df = self.fleet_df.copy()

        features = df[["SOH", "ambient_temperature", "cycle"]]

        model = IsolationForest(
            contamination=0.15,
            random_state=42
        )

        df["anomaly_score"] = model.fit_predict(features)
        df["is_anomaly"] = df["anomaly_score"] == -1

        anomalies = (
            df[df["is_anomaly"]]
            .sort_values("SOH")
            .drop(columns=["anomaly_score", "is_anomaly"])
        )
        anomalies["SOH"] = anomalies["SOH"].round(2)
        return anomalies

    def get_most_anomalous_battery(self) -> dict:
        anomalies = self.detect_anomalies()

        if anomalies.empty:
            return {
                "status": "No anomalies detected"
            }

        row = anomalies.iloc[0]

        return {
            "battery_id": row["battery_id"],
            "SOH": round(float(row["SOH"]), 2),
            "ambient_temperature": int(row["ambient_temperature"]),
            "cycle": int(row["cycle"]),
            "reason": "Battery behavior differs from the fleet pattern based on SOH, cycle, and temperature."
        }