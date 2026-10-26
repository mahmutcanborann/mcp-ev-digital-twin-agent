"""Early cycle life features for the Severson et al. (2019) LFP/graphite cells.

Task: early cycle life prediction - predict a cell's total cycle life (cycles
to 80% of nominal capacity) from its first 100 cycles. This module only
handles the Severson data; it is never combined with the NASA cells.

Features follow Table 1 of the paper. Cycle numbering matches the authors'
code: summary cycle k is the k-th summary row (cycle 1 is a formation /
empty cycle for batch 1), and Qdlin row j is cycles[j].

Cleaning: the raw summary data has isolated logging glitches (e.g. a 2.9 Ah
discharge capacity on a 1.1 Ah cell, zero internal resistance). Values that
deviate more than 10% from the centred 5-cycle rolling median are replaced
by that median before computing features.

Deviation from the paper: the summary data has no per-cycle time series of
temperature, so "integral of temperature over time, cycles 2-100" is
approximated by the sum of the per-cycle average temperature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Qdlin rows used for dQ_100-10(V) = Q_100(V) - Q_10(V), as in the authors' code
QDLIN_ROW_LATE = 99
QDLIN_ROW_EARLY = 9

DQ_FEATURES = ["dq_min", "dq_mean", "dq_var", "dq_skew", "dq_kurt", "dq_2v"]
FADE_FEATURES = ["fade_slope_2_100", "fade_intercept_2_100", "fade_slope_91_100", "fade_intercept_91_100",
                 "qd_cycle2", "qd_max_minus_cycle2", "qd_cycle100"]
OTHER_FEATURES = ["charge_time_first5", "t_max_2_100", "t_min_2_100", "t_avg_sum_2_100",
                  "ir_cycle2", "ir_min_2_100", "ir_100_minus_2"]
ALL_FEATURES = DQ_FEATURES + FADE_FEATURES + OTHER_FEATURES

# Feature sets of the paper's three models (Fig. 4)
PAPER_MODELS = {
    "variance": ["dq_var"],
    "discharge": ["dq_min", "dq_var", "dq_skew", "dq_kurt", "qd_cycle2", "qd_max_minus_cycle2"],
    "full": ["dq_var", "dq_min", "fade_slope_2_100", "fade_intercept_2_100", "qd_cycle2",
             "charge_time_first5", "t_avg_sum_2_100", "ir_min_2_100", "ir_100_minus_2"],
}


SUMMARY_SIGNALS = ["q_discharge", "ir", "t_max", "t_avg", "t_min", "charge_time"]
GLITCH_TOLERANCE = 0.10


def remove_glitches(summary: pd.DataFrame) -> pd.DataFrame:
    """Replace isolated logging glitches with the local rolling median."""
    out = summary.copy()
    for col in SUMMARY_SIGNALS:
        med = out[col].rolling(5, center=True, min_periods=1).median()
        bad = (out[col] - med).abs() > GLITCH_TOLERANCE * med.abs()
        out.loc[bad, col] = med[bad]
    return out


def _log_abs(x: float) -> float:
    return float(np.log10(np.abs(x)))


def _linfit(cycles: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(cycles, values, 1)
    return float(slope), float(intercept)


def cell_features(qdlin: np.ndarray, summary: pd.DataFrame) -> dict:
    """Features of one cell from its Qdlin curves (rows = cycles) and per-cycle summary."""
    dq = qdlin[QDLIN_ROW_LATE] - qdlin[QDLIN_ROW_EARLY]
    if np.isnan(dq).any():
        raise ValueError("missing Qd(V) curve for cycle 10 or 100")
    s = summary.set_index("cycle").sort_index()
    early = remove_glitches(s.loc[2:100])
    if len(early) < 99:
        raise ValueError("need summary data for cycles 2-100")
    qd = early["q_discharge"].to_numpy()
    cyc = early.index.to_numpy(dtype=float)
    slope, intercept = _linfit(cyc, qd)
    slope_late, intercept_late = _linfit(cyc[-10:], qd[-10:])
    return {
        "dq_min": _log_abs(dq.min()),
        "dq_mean": _log_abs(dq.mean()),
        "dq_var": _log_abs(dq.var()),
        "dq_skew": _log_abs(skew(dq)),
        "dq_kurt": _log_abs(kurtosis(dq)),
        "dq_2v": _log_abs(dq[-1]),  # the voltage grid runs from 3.5 V down to 2.0 V
        "fade_slope_2_100": slope,
        "fade_intercept_2_100": intercept,
        "fade_slope_91_100": slope_late,
        "fade_intercept_91_100": intercept_late,
        "qd_cycle2": float(qd[0]),
        "qd_max_minus_cycle2": float(qd.max() - qd[0]),
        "qd_cycle100": float(qd[-1]),
        "charge_time_first5": float(early["charge_time"].iloc[:5].mean()),
        "t_max_2_100": float(early["t_max"].max()),
        "t_min_2_100": float(early["t_min"].min()),
        "t_avg_sum_2_100": float(early["t_avg"].sum()),
        "ir_cycle2": float(early["ir"].iloc[0]),
        "ir_min_2_100": float(early["ir"].min()),
        "ir_100_minus_2": float(early["ir"].iloc[-1] - early["ir"].iloc[0]),
    }


def build_feature_table(cells: pd.DataFrame, summary: pd.DataFrame, qdlin: dict) -> pd.DataFrame:
    """One row per cell: id, split, cycle_life and all candidate features."""
    groups = dict(tuple(summary.groupby("cell_id")))
    rows = []
    for cell in cells.itertuples(index=False):
        feats = cell_features(np.asarray(qdlin[cell.cell_id], dtype=float), groups[cell.cell_id])
        rows.append({"cell_id": cell.cell_id, "batch": cell.batch, "split": cell.split,
                     "cycle_life": cell.cycle_life, **feats})
    return pd.DataFrame(rows)
