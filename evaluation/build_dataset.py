"""Build data/features/soh_dataset.csv from the raw NASA battery cycles.

Input (not in git, ~650 MB): Kaggle "NASA Battery Dataset", cleaned_dataset/
    data/metadata.csv
    data/data/00001.csv, 00002.csv, ...

Output (in git, small): data/features/soh_dataset.csv
    one row per discharge cycle: battery_id, cycle, ambient_temperature,
    capacity_ah, SOH (target) and partial-window features.

Usage:
    python evaluation/build_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.soh_features import extract_charge_features, extract_discharge_features  # noqa: E402

DATA_DIR = BASE_DIR / "data"
OUT = DATA_DIR / "features" / "soh_dataset.csv"


def main() -> None:
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    meta["order"] = np.arange(len(meta))
    raw_dir = DATA_DIR / "data"
    if not raw_dir.exists():
        sys.exit(f"Raw cycle files not found in {raw_dir}. See README > Data.")

    # ---- target: same definition as notebooks/battery_exploration.ipynb ----
    dis = meta[(meta["type"] == "discharge") & meta["Capacity"].notna()].copy()
    dis["capacity_ah"] = pd.to_numeric(dis["Capacity"], errors="coerce")
    dis["cycle"] = dis.groupby("battery_id").cumcount()
    dis["SOH"] = dis["capacity_ah"] / dis.groupby("battery_id")["capacity_ah"].transform("max") * 100
    dis = dis[dis["SOH"] > 20].copy()

    # ---- discharge features (first 300 s) ----
    rows = []
    for r in dis.itertuples():
        cycle = pd.read_csv(raw_dir / r.filename)
        rows.append({"order": r.order, **extract_discharge_features(cycle, r.ambient_temperature)})
    dis = dis.drop(columns=["ambient_temperature"]).merge(pd.DataFrame(rows), on="order")

    # ---- charge features from the most recent charge before each discharge ----
    chg_rows = []
    for r in meta[meta["type"] == "charge"].itertuples():
        cycle = pd.read_csv(raw_dir / r.filename)
        chg_rows.append({"chg_order": r.order, "battery_id": r.battery_id, **extract_charge_features(cycle)})
    chg = pd.DataFrame(chg_rows).sort_values("chg_order")

    dis = pd.merge_asof(dis.sort_values("order"), chg, left_on="order", right_on="chg_order",
                        by="battery_id", direction="backward")

    cols = ["battery_id", "cycle", "filename", "ambient_temperature", "capacity_ah", "SOH",
            "dis_i_mean", "dis_v_at_60s", "dis_v_at_300s", "dis_T_start",
            "chg_t_39_42", "chg_cv_t_to_05A", "chg_T_rise_cc"]
    out = dis[cols].sort_values(["battery_id", "cycle"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(BASE_DIR)}: {len(out)} discharge cycles, {out.battery_id.nunique()} batteries")


if __name__ == "__main__":
    main()
