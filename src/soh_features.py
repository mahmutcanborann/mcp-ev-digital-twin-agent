"""Feature extraction for measurement-based SOH estimation.

Only *partial* windows of a cycle are used, so the target (capacity) is never
leaked into the inputs:

* discharge: the first 5 minutes (300 s) of the discharge curve
* charge:    the 3.9 V -> 4.2 V part of the CC phase and the CV tail

A full discharge to cut-off voltage is deliberately NOT used: at constant
current, full-discharge time x current *is* the capacity, i.e. the label.
Real vehicles also rarely see full discharges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISCHARGE_WINDOW_S = 300.0

# Features used by the production SOH model (best in leave-one-battery-out evaluation)
DISCHARGE_FEATURES = [
    "ambient_temperature",
    "dis_i_mean",
    "dis_v_at_60s",
    "dis_v_at_300s",
    "dis_T_start",
]

CHARGE_FEATURES = [
    "chg_t_39_42",
    "chg_cv_t_to_05A",
    "chg_T_rise_cc",
]


def _interp_or_nan(t: np.ndarray, v: np.ndarray, at: float) -> float:
    if len(t) == 0 or t[-1] < at:
        return float("nan")
    return float(np.interp(at, t, v))


def extract_discharge_features(cycle: pd.DataFrame, ambient_temperature: float) -> dict:
    """Features from the first 300 s of one discharge cycle.

    `cycle` needs the NASA columns Time, Voltage_measured, Current_measured,
    Temperature_measured.
    """
    t = cycle["Time"].to_numpy(dtype=float)
    v = cycle["Voltage_measured"].to_numpy(dtype=float)
    i = cycle["Current_measured"].to_numpy(dtype=float)
    temp = cycle["Temperature_measured"].to_numpy(dtype=float)

    window = t <= DISCHARGE_WINDOW_S
    return {
        "ambient_temperature": float(ambient_temperature),
        "dis_i_mean": float(np.abs(i[window]).mean()) if window.any() else float("nan"),
        "dis_v_at_60s": _interp_or_nan(t, v, 60.0),
        "dis_v_at_300s": _interp_or_nan(t, v, 300.0),
        "dis_T_start": float(temp[0]) if len(temp) else float("nan"),
    }


def extract_charge_features(cycle: pd.DataFrame) -> dict:
    """Features from the 3.9-4.2 V window and the CV phase of one charge cycle."""
    t = cycle["Time"].to_numpy(dtype=float)
    v = cycle["Voltage_measured"].to_numpy(dtype=float)
    i = cycle["Current_measured"].to_numpy(dtype=float)
    temp = cycle["Temperature_measured"].to_numpy(dtype=float)

    def first_time(mask: np.ndarray) -> float:
        k = np.where(mask)[0]
        return float(t[k[0]]) if len(k) else float("nan")

    t39 = first_time(v >= 3.9)
    t42 = first_time(v >= 4.19)

    if np.isnan(t42):
        cv_time = float("nan")
        temp_rise = float("nan")
    else:
        k = np.where((t >= t42) & (i < 0.5))[0]
        cv_time = float(t[k[0]] - t42) if len(k) else float("nan")
        temp_rise = float(np.nanmax(temp[t <= t42]) - temp[0])

    return {
        "chg_t_39_42": t42 - t39,
        "chg_cv_t_to_05A": cv_time,
        "chg_T_rise_cc": temp_rise,
    }
