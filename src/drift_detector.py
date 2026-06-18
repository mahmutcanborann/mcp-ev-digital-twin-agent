import pandas as pd
from pathlib import Path


class DataDriftDetector:

    def __init__(self):
        base_dir = Path(__file__).resolve().parents[1]

        self.reference_df = pd.read_csv(
            base_dir / "models" / "fleet_latest_status.csv"
        )

    def detect_soh_drift(
        self,
        current_soh: float
    ) -> dict:


        reference_mean = self.reference_df["SOH"].mean()
        reference_std = self.reference_df["SOH"].std()

        drift = abs(
            current_soh - reference_mean
        )

        if drift < reference_std:
            status = "Normal"

        elif drift < 2 * reference_std:
            status = "Warning"

        else:
            status = "Critical"

        return {
            "reference_mean_soh":
                round(float(reference_mean), 2),

            "current_soh":
                round(float(current_soh), 2),

            "drift_distance":
                round(float(drift), 2),

            "drift_status":
                status,
            "reference_std_soh":
                round(float(reference_std), 2),
        }