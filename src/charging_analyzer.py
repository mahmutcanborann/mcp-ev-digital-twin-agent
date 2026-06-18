import pandas as pd
from pathlib import Path


class ChargingAnalyzer:

    def __init__(self):
        base_dir = Path(__file__).resolve().parents[1]

        self.df = pd.read_csv(
            base_dir / "models" / "charging_risk_features.csv"
        )

    def get_summary(self):
        return {
            "avg_duration": float(round(
                self.df["charging_duration_hours"].mean(),
                2
            )),
            "avg_soc_increase": float(round(
                self.df["soc_increase"].mean(),
                2
            )),
            "fast_charge_ratio": float(round(
                self.df["is_fast_charge"].mean() * 100,
                2
            )),
            "avg_risk_score": float(round(
                self.df["charging_risk_score"].mean(),
                2
            ))
        }

    def risk_distribution(self):

        return (
            self.df["charging_risk_level"]
            .value_counts()
            .to_dict()
        )

    def high_risk_sessions(self):

        return (
            self.df[
                self.df["charging_risk_level"] == "High"
            ]
            .head(10)
            .to_dict("records")
        )