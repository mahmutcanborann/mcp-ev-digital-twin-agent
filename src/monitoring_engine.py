import pandas as pd
from pathlib import Path
from src.ev_twin_report import EVTwinReport


class MonitoringEngine:
    def __init__(self):
        self.report_engine = EVTwinReport()

        base_dir = Path(__file__).resolve().parents[1]
        self.fleet_df = pd.read_csv(
            base_dir / "models" / "fleet_latest_status.csv"
        )

    def check_system_health(
        self,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> dict:
        report = self.report_engine.generate_report(
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

        warnings = []

        battery_soh = report["battery"]["predicted_soh"]
        charging_risk = report["charging"]["summary"]["avg_risk_score"]
        driving_score = report["driving"]["driving_score"]
        overall_score = report["overall_ev_twin_score"]

        if battery_soh < 80:
            warnings.append("Battery SOH is below 80% threshold.")

        if charging_risk > 40:
            warnings.append("Charging risk score is elevated.")

        if driving_score < 60:
            warnings.append("Driving score indicates aggressive behavior.")

        if overall_score < 60:
            warnings.append("Overall EV Twin score is low.")

        if not warnings:
            system_status = "Healthy"
        elif len(warnings) <= 2:
            system_status = "Warning"
        else:
            system_status = "Critical"

        return {
            "system_status": system_status,
            "battery_soh": battery_soh,
            "charging_risk_score": charging_risk,
            "driving_score": driving_score,
            "overall_ev_twin_score": overall_score,
            "warnings": warnings
        }

    def monitor_fleet(self) -> dict:
        fleet_size = len(self.fleet_df)
        avg_soh = round(float(self.fleet_df["SOH"].mean()), 2)

        critical = self.fleet_df[self.fleet_df["SOH"] < 70]
        warning = self.fleet_df[
            (self.fleet_df["SOH"] >= 70) &
            (self.fleet_df["SOH"] < 80)
        ]
        healthy = self.fleet_df[self.fleet_df["SOH"] >= 80]

        worst = self.fleet_df.sort_values("SOH").iloc[0]
        best = self.fleet_df.sort_values("SOH", ascending=False).iloc[0]

        if len(critical) > 0:
            fleet_status = "Critical"
        elif len(warning) > 0:
            fleet_status = "Warning"
        else:
            fleet_status = "Healthy"

        return {
            "fleet_status": fleet_status,
            "fleet_size": int(fleet_size),
            "average_soh": avg_soh,
            "critical_batteries": int(len(critical)),
            "warning_batteries": int(len(warning)),
            "healthy_batteries": int(len(healthy)),
            "worst_battery": {
                "battery_id": worst["battery_id"],
                "soh": round(float(worst["SOH"]), 2),
                "cycle": int(worst["cycle"])
            },
            "best_battery": {
                "battery_id": best["battery_id"],
                "soh": round(float(best["SOH"]), 2),
                "cycle": int(best["cycle"])
            }
        }