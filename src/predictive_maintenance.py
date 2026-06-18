from src.ev_twin_report import EVTwinReport


class PredictiveMaintenanceEngine:

    def __init__(self):
        self.report_engine = EVTwinReport()

    def predict_actions(
        self,
        battery_id: str,
        cycle: int,
        ambient_temperature: int = 24
    ) -> dict:

        report = self.report_engine.generate_report(
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=ambient_temperature
        )

        actions = []

        soh = report["battery"]["predicted_soh"]

        if soh < 80:
            actions.append(
                "Monitor battery degradation monthly"
            )

        if soh < 70:
            actions.append(
                "Schedule battery inspection"
            )

        fast_charge_ratio = report["charging"]["summary"][
            "fast_charge_ratio"
        ]

        if fast_charge_ratio > 30:
            actions.append(
                "Reduce DC fast charging frequency"
            )

        avg_risk_score = report["charging"]["summary"][
            "avg_risk_score"
        ]

        if avg_risk_score > 40:
            actions.append(
                "Avoid charging above 90% SOC"
            )

        driving_style = report["driving"][
            "driving_style"
        ]

        if driving_style == "Aggressive":
            actions.append(
                "Adopt smoother acceleration and braking"
            )

        overall_score = report[
            "overall_ev_twin_score"
        ]

        if overall_score >= 80:
            priority = "Low"

        elif overall_score >= 60:
            priority = "Medium"

        else:
            priority = "High"

        return {
            "priority": priority,
            "overall_score": overall_score,
            "recommended_actions": actions
        }