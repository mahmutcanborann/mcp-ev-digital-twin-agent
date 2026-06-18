import requests

from src.ev_twin_report import EVTwinReport
from src.predictive_maintenance import PredictiveMaintenanceEngine
from src.monitoring_engine import MonitoringEngine


class LLMEVDigitalTwinAgent:
    def __init__(self, model_name="gemma3:4b"):
        self.model_name = model_name
        self.report_engine = EVTwinReport()
        self.maintenance_engine = PredictiveMaintenanceEngine()
        self.monitoring_engine = MonitoringEngine()

    def ask(
        self,
        question: str,
        battery_id: str = "B0005",
        cycle: int = 150,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> str:
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

        system_health = self.monitoring_engine.check_system_health(
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=ambient_temperature,
            nominal_range_km=nominal_range_km
        )

        fleet_health = self.monitoring_engine.monitor_fleet()

        prompt = f"""
You are an EV Digital Twin AI assistant.

Use ONLY the data below. Do not invent values.

User question:
{question}

Selected vehicle/battery:
- Battery ID: {battery_id}
- Cycle: {cycle}
- Ambient temperature: {ambient_temperature} C
- Nominal range: {nominal_range_km} km

Battery data:
- SOH: {report["battery"]["predicted_soh"]}%
- Estimated range: {report["battery"]["estimated_range_km"]} km
- Health status: {report["battery"]["health_status"]}
- RUL to 80% SOH: {report["battery"]["remaining_cycles_until_80_soh"]} cycles

Charging data:
- Average charging duration: {report["charging"]["summary"]["avg_duration"]} h
- Fast charging usage: {report["charging"]["summary"]["fast_charge_ratio"]}%
- Average charging risk score: {report["charging"]["summary"]["avg_risk_score"]}

Driving data:
- Driving style: {report["driving"]["driving_style"]}
- Driving score: {report["driving"]["driving_score"]}/100
- Average speed: {report["driving"]["summary"]["avg_speed"]} km/h

Maintenance:
- Priority: {maintenance["priority"]}
- Recommended actions: {maintenance["recommended_actions"]}

Monitoring:
- System status: {system_health["system_status"]}
- Warnings: {system_health["warnings"]}
- Fleet status: {fleet_health["fleet_status"]}
- Fleet average SOH: {fleet_health["average_soh"]}%
- Critical batteries: {fleet_health["critical_batteries"]}

Answer clearly and professionally in 4-6 sentences.
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"]