"""Build the early cycle life feature table for the Severson et al. (2019) cells.

Reads data/severson/processed/{cells.csv, summary.csv.gz, qdlin_first100.npz}
(made by build_severson_dataset.py) and writes
data/severson/processed/features.csv: one row per cell with its split, cycle
life and the 20 candidate features of the paper's Table 1. features.csv is
in git, so the evaluation runs without the 42 MB Qd(V) file.

Usage:
    python evaluation/build_cycle_life_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.cycle_life_features import build_feature_table  # noqa: E402

DATA = BASE_DIR / "data" / "severson" / "processed"


def main() -> None:
    cells = pd.read_csv(DATA / "cells.csv")
    summary = pd.read_csv(DATA / "summary.csv.gz")
    with np.load(DATA / "qdlin_first100.npz") as qdlin:
        table = build_feature_table(cells, summary, qdlin)
    table.round(8).to_csv(DATA / "features.csv", index=False)
    print(f"wrote {DATA / 'features.csv'}: {len(table)} cells")


if __name__ == "__main__":
    main()
