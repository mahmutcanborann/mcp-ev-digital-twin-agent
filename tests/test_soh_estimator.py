import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.soh_features import DISCHARGE_FEATURES, extract_discharge_features  # noqa: E402
from src.soh_estimator import SOHEstimator  # noqa: E402


def synthetic_discharge(duration_s=3000, v0=4.2, slope=0.0004, current=-2.0):
    t = np.arange(0, duration_s, 10.0)
    return pd.DataFrame({
        "Time": t,
        "Voltage_measured": v0 - slope * t,
        "Current_measured": np.full_like(t, current),
        "Temperature_measured": 24 + 0.002 * t,
    })


def test_discharge_features_use_only_first_5_minutes():
    f = extract_discharge_features(synthetic_discharge(), ambient_temperature=24)
    assert f["dis_v_at_60s"] == pytest.approx(4.2 - 0.0004 * 60)
    assert f["dis_v_at_300s"] == pytest.approx(4.2 - 0.0004 * 300)
    assert f["dis_i_mean"] == pytest.approx(2.0)
    assert f["dis_T_start"] == pytest.approx(24.0)


def test_short_discharge_gives_nan_instead_of_guessing():
    f = extract_discharge_features(synthetic_discharge(duration_s=200), ambient_temperature=24)
    assert np.isnan(f["dis_v_at_300s"])


def test_no_target_leakage_in_features():
    # capacity / full-discharge duration are the label; they must never be inputs
    forbidden = ("capacity", "duration", "SOH", "cycle", "battery")
    assert not any(word.lower() in f.lower() for f in DISCHARGE_FEATURES for word in forbidden)


@pytest.fixture(scope="module")
def estimator():
    return SOHEstimator()


def test_estimate_is_out_of_sample_and_close_for_a_reference_battery(estimator):
    r = estimator.estimate_for_battery_cycle("B0005", 100)
    assert r["cycle"] == 100
    assert 0 < r["estimated_soh"] < 110
    assert r["absolute_error"] == pytest.approx(abs(r["estimated_soh"] - r["measured_soh"]), abs=0.02)
    assert "without this battery" in r["method"]
    assert r["absolute_error"] < 10


def test_health_status_is_returned_with_twin_thresholds(estimator):
    from src.soh_estimator import health_status
    assert [health_status(s) for s in (95, 84, 75, 60)] == ["Excellent", "Good", "Moderate", "Degraded"]
    r = estimator.estimate_for_battery_cycle("B0018", 60)
    assert r["health_status"] == health_status(r["estimated_soh"])


def test_validation_block_uses_exact_tool_numbers(estimator):
    import json
    from agent.discovery_mcp_agent import DiscoveryMCPAgent
    r = estimator.estimate_for_battery_cycle("B0018", 60)
    block = DiscoveryMCPAgent.format_validation(json.dumps(r))
    assert f"Measured SOH: {r['measured_soh']}%" in block
    assert f"Estimation Error: {r['absolute_error']} SOH points" in block
    # other tools (no measured SOH) get no validation block
    assert DiscoveryMCPAgent.format_validation(json.dumps({"predicted_soh": 80})) == ""
    assert DiscoveryMCPAgent.format_validation("not json") == ""


def test_unknown_battery_is_reported_not_crashed(estimator):
    r = estimator.estimate_for_battery_cycle("B9999", 10)
    assert "error" in r and "B0005" in r["available_batteries"]


def test_estimate_from_raw_cycle(estimator):
    r = estimator.estimate_from_cycle(synthetic_discharge(), ambient_temperature=24)
    assert 0 < r["estimated_soh"] < 110
    assert set(r["features"]) == set(DISCHARGE_FEATURES)
