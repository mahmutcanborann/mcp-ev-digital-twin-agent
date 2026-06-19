import pandas as pd
import joblib
from pathlib import Path


class BatteryDigitalTwin:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[1]

        model_path = base_dir / "models" / "battery_soh_rf_model.pkl"
        encoder_path = base_dir / "models" / "battery_label_encoder.pkl"

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

    def predict_soh(
        self,
        battery_id: str,
        cycle: int,
        ambient_temperature: int = 24
    ) -> float:
        battery_code = self.encoder.transform([battery_id])[0]

        X = pd.DataFrame({
            "battery_code": [battery_code],
            "ambient_temperature": [ambient_temperature],
            "cycle": [cycle]
        })

        pred = self.model.predict(X)[0]

        return round(float(pred), 2)

    def estimate_rul(
        self,
        battery_id: str,
        current_cycle: int,
        threshold: float = 80,
        ambient_temperature: int = 24,
        max_cycle: int = 500
    ) -> int:
        for c in range(current_cycle, max_cycle + 1):
            soh = self.predict_soh(
                battery_id=battery_id,
                cycle=c,
                ambient_temperature=ambient_temperature
            )

            if soh <= threshold:
                return c - current_cycle

        return -1

    def estimate_range(
        self,
        soh: float,
        nominal_range_km: float = 500
    ) -> float:
        estimated_range = nominal_range_km * (soh / 100)
        return round(estimated_range, 2)

    def simulate_future(
        self,
        battery_id: str,
        start_cycle: int,
        end_cycle: int,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> list:
        results = []

        for c in range(start_cycle, end_cycle + 1):
            soh = self.predict_soh(
                battery_id=battery_id,
                cycle=c,
                ambient_temperature=ambient_temperature
            )

            estimated_range = self.estimate_range(
                soh=soh,
                nominal_range_km=nominal_range_km
            )

            results.append({
                "cycle": c,
                "predicted_soh": soh,
                "estimated_range_km": estimated_range
            })

        return results
    def generate_timeline(
        self,
        battery_id: str,
        start_cycle: int = 0,
        end_cycle: int = 200,
        step: int = 50,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> list:
        timeline = []

        for c in range(start_cycle, end_cycle + 1, step):
            soh = self.predict_soh(
                battery_id=battery_id,
                cycle=c,
                ambient_temperature=ambient_temperature
            )

            estimated_range = self.estimate_range(
                soh=soh,
                nominal_range_km=nominal_range_km
            )

            timeline.append({
                "battery_id": battery_id,
                "cycle": c,
                "predicted_soh": soh,
                "estimated_range_km": estimated_range,
                "health_status": self.get_health_status(soh)
            })

        return timeline
    def get_health_status(self, soh: float) -> str:
        if soh >= 90:
            return "Excellent"
        elif soh >= 80:
            return "Good"
        elif soh >= 70:
            return "Moderate"
        else:
            return "Degraded"

    def generate_summary(
        self,
        battery_id: str,
        cycle: int,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> dict:
        soh = self.predict_soh(
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=ambient_temperature
        )

        rul = self.estimate_rul(
            battery_id=battery_id,
            current_cycle=cycle,
            threshold=80,
            ambient_temperature=ambient_temperature
        )

        estimated_range = self.estimate_range(
            soh=soh,
            nominal_range_km=nominal_range_km
        )

        status = self.get_health_status(soh)

        return {
            "battery_id": battery_id,
            "cycle": cycle,
            "ambient_temperature": ambient_temperature,
            "predicted_soh": soh,
            "estimated_range_km": estimated_range,
            "remaining_cycles_until_80_soh": rul,
            "health_status": status
        }

    def simulate_what_if_scenarios(
        self,
        battery_id: str,
        start_cycle: int,
        end_cycle: int,
        ambient_temperature: int = 24,
        nominal_range_km: float = 500
    ) -> dict:
            scenarios = {
                "Normal": {
                    "temperature": ambient_temperature,
                    "extra_degradation": 0.00
                },
                "Cold Weather": {
                    "temperature": 4,
                    "extra_degradation": 0.02
                },
                "Hot Weather": {
                    "temperature": 44,
                    "extra_degradation": 0.04
                },
                "Aggressive Usage": {
                    "temperature": ambient_temperature,
                    "extra_degradation": 0.08
                }
            }

            all_results = {}

            for scenario_name, config in scenarios.items():
                scenario_results = []

                baseline_soh = self.predict_soh(
                    battery_id=battery_id,
                    cycle=start_cycle,
                    ambient_temperature=config["temperature"]
                )

                for c in range(start_cycle, end_cycle + 1):
                    predicted_soh = self.predict_soh(
                        battery_id=battery_id,
                        cycle=c,
                        ambient_temperature=config["temperature"]
                    )

                    cycle_delta = c - start_cycle

                    adjusted_soh = predicted_soh - (
                        cycle_delta * config["extra_degradation"]
                    )

                    adjusted_soh = max(adjusted_soh, 0)

                    estimated_range = self.estimate_range(
                        soh=adjusted_soh,
                        nominal_range_km=nominal_range_km
                    )

                    scenario_results.append({
                        "scenario": scenario_name,
                        "cycle": c,
                        "predicted_soh": round(adjusted_soh, 2),
                        "estimated_range_km": estimated_range,
                        "temperature": config["temperature"],
                        "baseline_soh": baseline_soh
                    })

                all_results[scenario_name] = scenario_results

            return all_results