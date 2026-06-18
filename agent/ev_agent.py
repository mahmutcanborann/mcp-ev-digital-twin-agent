import pandas as pd
from pathlib import Path
from src.ev_twin_report import EVTwinReport
from src.predictive_maintenance import PredictiveMaintenanceEngine


class EVDigitalTwinAgent:
    def __init__(self):
        self.report_engine = EVTwinReport()
        self.maintenance_engine = PredictiveMaintenanceEngine()
        base_dir = Path(__file__).resolve().parents[1]
        self.fleet_df = pd.read_csv(
            base_dir / "models" / "fleet_latest_status.csv"
        )

    def ask(
        self,
        question: str,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        q = question.lower()

        if "compare" in q:
            return self.compare_batteries(
                question=question,
                cycle=cycle,
                ambient_temperature=ambient_temperature,
                nominal_range_km=nominal_range_km
            )

        if "riskiest" in q or "worst" in q:
            return self.get_riskiest_battery()

        if "healthiest" in q or "best" in q:
            return self.get_healthiest_battery()

        report = self.report_engine.generate_report(
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

        maintenance = self.maintenance_engine.predict_actions(
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=ambient_temperature
        )

        if "health" in q or "soh" in q or "battery" in q:
            return (
                f"Battery {battery_id} has an estimated SOH of "
                f"{report['battery']['predicted_soh']}%. "
                f"Health status: {report['battery']['health_status']}. "
                f"Estimated range: {report['battery']['estimated_range_km']} km."
            )

        if "range" in q:
            return (
                f"The estimated range for battery {battery_id} is "
                f"{report['battery']['estimated_range_km']} km "
                f"based on SOH {report['battery']['predicted_soh']}%."
            )

        if "maintenance" in q or "action" in q or "recommend" in q:
            actions = maintenance["recommended_actions"]

            if not actions:
                return "No critical maintenance action is currently required."

            return (
                f"Maintenance priority is {maintenance['priority']}. "
                f"Recommended actions: " + "; ".join(actions)
            )

        if "charging" in q or "charge" in q:
            charging = report["charging"]["summary"]

            return (
                f"Average charging duration is {charging['avg_duration']} hours. "
                f"Fast charging usage is {charging['fast_charge_ratio']}%. "
                f"Average charging risk score is {charging['avg_risk_score']}."
            )

        if "driving" in q or "drive" in q:
            driving = report["driving"]

            return (
                f"Driving style is {driving['driving_style']}. "
                f"Driving score is {driving['driving_score']}/100. "
                f"Average speed is {driving['summary']['avg_speed']} km/h."
            )

        if "overall" in q or "score" in q:
            return (
                f"Overall EV Twin Score is "
                f"{report['overall_ev_twin_score']}/100. "
                f"{report['recommendation']}"
            )

        return (
            f"Overall EV Twin Score is {report['overall_ev_twin_score']}/100. "
            f"Battery SOH is {report['battery']['predicted_soh']}%. "
            f"Driving style is {report['driving']['driving_style']}. "
            f"Recommendation: {report['recommendation']}"
        )

    def compare_batteries(
        self,
        question: str,
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
        import re

        battery_ids = re.findall(r"B\d{4}", question.upper())

        if len(battery_ids) < 2:
            return "Please provide two battery IDs to compare, for example: Compare B0005 and B0047."

        b1, b2 = battery_ids[0], battery_ids[1]

        r1 = self.report_engine.generate_report(
            battery_id=b1,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

        r2 = self.report_engine.generate_report(
            battery_id=b2,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

        soh1 = r1["battery"]["predicted_soh"]
        soh2 = r2["battery"]["predicted_soh"]

        range1 = r1["battery"]["estimated_range_km"]
        range2 = r2["battery"]["estimated_range_km"]

        status1 = r1["battery"]["health_status"]
        status2 = r2["battery"]["health_status"]

        if soh1 > soh2:
            better = b1
        elif soh2 > soh1:
            better = b2
        else:
            better = "Both batteries are similar"

        return (
            f"{b1}: SOH {soh1}%, range {range1} km, status {status1}. "
            f"{b2}: SOH {soh2}%, range {range2} km, status {status2}. "
            f"Better battery: {better}."
        )


    def get_riskiest_battery(self) -> str:
        row = self.fleet_df.sort_values(
            "SOH",
            ascending=True
        ).iloc[0]

        return (
            f"The riskiest battery is {row['battery_id']} "
            f"with SOH {row['SOH']:.2f}% at cycle {int(row['cycle'])}."
        )

    def get_healthiest_battery(self) -> str:
        row = self.fleet_df.sort_values(
            "SOH",
            ascending=False
        ).iloc[0]

        return (
            f"The healthiest battery is {row['battery_id']} "
            f"with SOH {row['SOH']:.2f}% at cycle {int(row['cycle'])}."
        )