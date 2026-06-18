import pandas as pd
from pathlib import Path


class DrivingAnalyzer:

    def __init__(self):

        base_dir = Path(__file__).resolve().parents[1]

        self.df = pd.read_csv(
            base_dir / "models" / "electric-vehicle-telemetry-dataset.csv"
        )

        self.df["POWER_W"] = (
            self.df["VOL"] *
            self.df["CUR"]
        )

        self.df["POWER_KW"] = (
            self.df["POWER_W"] / 1000
        )

    def get_energy_summary(self):

        return {
            "avg_speed": round(
                float(self.df["SPD"].mean()),
                2
            ),

            "avg_power_kw": round(
                float(self.df["POWER_KW"].mean()),
                2
            ),

            "max_power_kw": round(
                float(self.df["POWER_KW"].max()),
                2
            )
        }

    def get_driving_score(self):

        high_speed_ratio = (
            (self.df["SPD"] > 100)
            .mean() * 100
        )

        hard_brake_ratio = (
            (self.df["BRK"] > 10)
            .mean() * 100
        )

        avg_acc = self.df["ACC"].mean()

        score = 100

        score -= high_speed_ratio * 2
        score -= hard_brake_ratio * 3
        score -= avg_acc * 0.5

        score = max(
            0,
            min(100, score)
        )

        return round(score, 2)

    def get_driving_style(self):

        score = self.get_driving_score()

        if score >= 80:
            return "Eco"

        elif score >= 60:
            return "Normal"

        else:
            return "Aggressive"