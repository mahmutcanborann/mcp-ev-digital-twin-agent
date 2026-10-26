"""Measurement-based SOH estimator.

Estimates battery State of Health from the first 5 minutes of a discharge
curve (see src/soh_features.py). Unlike the original `BatteryDigitalTwin`
model, it does not take a battery ID as input, so it works for batteries it
has never seen. Validation: leave-one-battery-out, see evaluation/results.md.

The model is small (gradient boosting on ~2.3k rows) and is trained from
data/features/soh_dataset.csv on start-up, which avoids pickle/scikit-learn
version issues across machines.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from src.soh_features import DISCHARGE_FEATURES, extract_discharge_features
except ImportError:  # running from inside src/
    from soh_features import DISCHARGE_FEATURES, extract_discharge_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "features" / "soh_dataset.csv"
MIN_CYCLES_PER_BATTERY = 40

# Leave-one-battery-out MAE on batteries with >= 40 cycles (evaluation/results.md)
VALIDATED_MAE_SOH_POINTS = 4.8


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)


class SOHEstimator:
    def __init__(self, dataset_path: Path | str = DATASET_PATH):
        df = pd.read_csv(dataset_path)
        n_cycles = df.groupby("battery_id")["cycle"].transform("size")
        self.data = df[n_cycles >= MIN_CYCLES_PER_BATTERY].reset_index(drop=True)
        self.features = list(DISCHARGE_FEATURES)
        self.model = make_model().fit(self.data[self.features], self.data["SOH"])
        self._holdout_models: dict[str, HistGradientBoostingRegressor] = {}

    # ------------------------------------------------------------------ core
    def estimate_from_features(self, features: dict) -> float:
        x = pd.DataFrame([{f: features.get(f) for f in self.features}], dtype=float)
        return round(float(self.model.predict(x)[0]), 2)

    def estimate_from_cycle(self, cycle: pd.DataFrame, ambient_temperature: float) -> dict:
        """Estimate SOH from a raw discharge cycle (NASA column format)."""
        feats = extract_discharge_features(cycle, ambient_temperature)
        return {
            "estimated_soh": self.estimate_from_features(feats),
            "features": {k: round(v, 4) for k, v in feats.items()},
            "typical_error_soh_points": VALIDATED_MAE_SOH_POINTS,
        }

    # ------------------------------------------------------- demo on NASA data
    def _model_without(self, battery_id: str) -> HistGradientBoostingRegressor:
        if battery_id not in self._holdout_models:
            train = self.data[self.data["battery_id"] != battery_id]
            self._holdout_models[battery_id] = make_model().fit(train[self.features], train["SOH"])
        return self._holdout_models[battery_id]

    def estimate_for_battery_cycle(self, battery_id: str, cycle: int) -> dict:
        """Estimate SOH for a recorded NASA test cycle, with a model trained WITHOUT that battery.

        Returns the measured SOH next to the estimate so the error is visible.
        """
        battery_id = battery_id.upper()
        rows = self.data[self.data["battery_id"] == battery_id]
        if rows.empty:
            known = sorted(self.data["battery_id"].unique())
            return {"error": f"Unknown battery '{battery_id}'.", "available_batteries": known}

        nearest = rows.iloc[(rows["cycle"] - cycle).abs().argsort().iloc[0]]
        x = pd.DataFrame([nearest[self.features]], dtype=float)
        estimate = float(self._model_without(battery_id).predict(x)[0])
        measured = float(nearest["SOH"])

        result = {
            "battery_id": battery_id,
            "cycle": int(nearest["cycle"]),
            "estimated_soh": round(estimate, 2),
            "measured_soh": round(measured, 2),
            "absolute_error": round(abs(estimate - measured), 2),
            "method": "first 5 minutes of the discharge curve; model trained without this battery",
            "typical_error_soh_points": VALIDATED_MAE_SOH_POINTS,
        }
        if int(nearest["cycle"]) != int(cycle):
            result["note"] = f"Cycle {cycle} not available; used nearest recorded cycle {int(nearest['cycle'])}."
        return result
