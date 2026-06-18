from src.battery_twin import BatteryDigitalTwin
from src.charging_analyzer import ChargingAnalyzer
from src.driving_analyzer import DrivingAnalyzer


class EVTwinReport:
    def __init__(self):
        self.battery_twin = BatteryDigitalTwin()
        self.charging_analyzer = ChargingAnalyzer()
        self.driving_analyzer = DrivingAnalyzer()

    def generate_report(
        self,
        battery_id: str,
        cycle: int,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> dict:
        battery_summary = self.battery_twin.generate_summary(
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

        charging_summary = self.charging_analyzer.get_summary()
        charging_risk_distribution = self.charging_analyzer.risk_distribution()

        driving_summary = self.driving_analyzer.get_energy_summary()
        driving_score = float(self.driving_analyzer.get_driving_score())
        driving_style = self.driving_analyzer.get_driving_style()

        battery_score = battery_summary["predicted_soh"]
        charging_score = max(
            0,
            100 - charging_summary["avg_risk_score"]
        )
        overall_score = float(round(
        (battery_score * 0.5) +
        (charging_score * 0.25) +
        (driving_score * 0.25),
        2
))

        recommendation = self.generate_recommendation(
            battery_summary=battery_summary,
            charging_summary=charging_summary,
            driving_style=driving_style,
            overall_score=overall_score
        )

        return {
            "overall_ev_twin_score": overall_score,
            "battery": battery_summary,
            "charging": {
                "summary": charging_summary,
                "risk_distribution": charging_risk_distribution
            },
            "driving": {
                "summary": driving_summary,
                "driving_score": driving_score,
                "driving_style": driving_style
            },
            "recommendation": recommendation
        }

    def generate_recommendation(
        self,
        battery_summary: dict,
        charging_summary: dict,
        driving_style: str,
        overall_score: float
    ) -> str:
        recommendations = []

        if battery_summary["predicted_soh"] < 80:
            recommendations.append(
                "Battery SOH is below 80%; monitor degradation and plan maintenance."
            )

        if charging_summary["fast_charge_ratio"] > 30:
            recommendations.append(
                "Fast charging usage is relatively high; reduce frequent DC fast charging."
            )

        if charging_summary["avg_risk_score"] > 40:
            recommendations.append(
                "Charging behavior shows medium/high risk; avoid high SOC and high-temperature charging."
            )

        if driving_style == "Aggressive":
            recommendations.append(
                "Driving behavior is aggressive; smoother acceleration and braking can improve efficiency."
            )

        if overall_score >= 80:
            recommendations.append(
                "Overall EV twin status is good."
            )
        elif overall_score >= 60:
            recommendations.append(
                "Overall EV twin status is moderate; optimization is recommended."
            )
        else:
            recommendations.append(
                "Overall EV twin status is risky; battery, charging, and driving behavior should be reviewed."
            )

        return " ".join(recommendations)