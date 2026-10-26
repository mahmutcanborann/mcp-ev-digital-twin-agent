"""Tests for the Severson early cycle life features and evaluation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from evaluation.evaluate_cycle_life import PAPER_MODELS, PAPER_RESULTS, evaluate, make_elastic_net  # noqa: E402
from src.cycle_life_features import ALL_FEATURES, cell_features, remove_glitches  # noqa: E402

FEATURES = BASE_DIR / "data" / "severson" / "processed" / "features.csv"


def synthetic_cell(fade_per_cycle=1e-5, glitch=False):
    cycles = np.arange(1, 121, dtype=float)
    summary = pd.DataFrame({
        "cycle": cycles,
        "q_discharge": 1.07 - fade_per_cycle * cycles,
        "ir": np.full_like(cycles, 0.016),
        "t_max": np.full_like(cycles, 36.0),
        "t_avg": np.full_like(cycles, 32.0),
        "t_min": np.full_like(cycles, 30.0),
        "charge_time": np.full_like(cycles, 10.0),
    })
    if glitch:
        summary.loc[summary.cycle == 40, "q_discharge"] = 2.9
        summary.loc[summary.cycle == 50, "ir"] = 0.0
    voltage = np.linspace(3.5, 2.0, 1000)
    q_curve = 1.07 * (3.5 - voltage) / 1.5
    qdlin = np.vstack([q_curve * (1 - 1e-4 * j) for j in range(101)])
    return qdlin, summary


def test_cell_features_cover_all_candidates():
    feats = cell_features(*synthetic_cell())
    assert set(feats) == set(ALL_FEATURES)
    assert all(np.isfinite(v) for v in feats.values())


def test_fade_slope_matches_linear_fade():
    feats = cell_features(*synthetic_cell(fade_per_cycle=2e-5))
    assert feats["fade_slope_2_100"] == pytest.approx(-2e-5, rel=1e-6)
    assert feats["qd_cycle2"] == pytest.approx(1.07 - 4e-5)


def test_glitches_do_not_change_features():
    clean = cell_features(*synthetic_cell())
    noisy = cell_features(*synthetic_cell(glitch=True))
    assert noisy["qd_max_minus_cycle2"] == pytest.approx(clean["qd_max_minus_cycle2"])
    assert noisy["ir_min_2_100"] == pytest.approx(clean["ir_min_2_100"])


def test_remove_glitches_keeps_smooth_signal():
    _, summary = synthetic_cell()
    pd.testing.assert_frame_equal(remove_glitches(summary), summary)


def test_missing_curve_raises():
    qdlin, summary = synthetic_cell()
    qdlin[99] = np.nan
    with pytest.raises(ValueError):
        cell_features(qdlin, summary)


@pytest.fixture(scope="module")
def features():
    return pd.read_csv(FEATURES)


def test_feature_table_uses_paper_split(features):
    assert len(features) == 124
    assert features.split.value_counts().to_dict() == {"test": 43, "train": 41, "secondary_test": 40}
    assert features[ALL_FEATURES].notna().all().all()


def test_variance_feature_correlates_with_cycle_life(features):
    # paper: rho = -0.93 between log var(dQ_100-10) and log cycle life
    rho = np.corrcoef(features["dq_var"], np.log10(features["cycle_life"]))[0, 1]
    assert rho < -0.9


def test_variance_model_reproduces_paper(features):
    res, _ = evaluate(features, PAPER_MODELS["variance"], make_elastic_net)
    for split in ("train", "test", "secondary_test"):
        assert res[split]["pct"] == pytest.approx(PAPER_RESULTS["variance"]["pct"][split], abs=1.0)


def test_models_beat_training_mean(features):
    baseline, _ = evaluate(features, None, None)
    model, _ = evaluate(features, PAPER_MODELS["discharge"], make_elastic_net)
    assert model["all_test_no_outlier"]["pct"] < baseline["all_test_no_outlier"]["pct"] / 2
