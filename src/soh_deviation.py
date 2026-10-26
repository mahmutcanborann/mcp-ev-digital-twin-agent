"""Fleet SOH deviation check.

Compares one battery's SOH with the fleet's mean and standard deviation
(z-score style outlier check).

This is deliberately NOT called "data drift detection": drift monitoring
compares the *distribution* of model inputs/outputs between training and
production data over time. With 34 lab cells and no production stream, a
per-battery deviation check is the honest thing this project can offer.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]


class FleetSOHDeviationChecker:
    def __init__(self, fleet_path: Path | str = BASE_DIR / "models" / "fleet_latest_status.csv"):
        self.reference_df = pd.read_csv(fleet_path)
        self.reference_mean = float(self.reference_df["SOH"].mean())
        self.reference_std = float(self.reference_df["SOH"].std())

    def check(self, current_soh: float) -> dict:
        """Status: Normal (< 1 std from fleet mean), Warning (< 2 std), Outlier (>= 2 std)."""
        deviation = float(current_soh) - self.reference_mean
        z_score = deviation / self.reference_std if self.reference_std > 0 else 0.0

        if abs(z_score) < 1:
            status = "Normal"
        elif abs(z_score) < 2:
            status = "Warning"
        else:
            status = "Outlier"

        return {
            "current_soh": round(float(current_soh), 2),
            "fleet_mean_soh": round(self.reference_mean, 2),
            "fleet_std_soh": round(self.reference_std, 2),
            "deviation_soh_points": round(deviation, 2),
            "z_score": round(z_score, 2),
            "status": status,
            "method": "z-score against fleet mean/std (outlier check, not distribution drift)",
        }
