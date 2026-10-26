"""Early cycle life prediction on the Severson et al. (2019) cells.

Predict total cycle life (cycles to 80% of nominal capacity) from the first
100 cycles, and compare with the paper's published numbers. Separate from
the NASA SOH evaluation: different chemistry, protocol, model and metrics.

Writes evaluation/cycle_life_results.md and evaluation/cycle_life_results.json.

Protocol (same as the paper)
----------------------------
* Split from cells.csv: 41 train / 43 primary test (batches 1+2 interleaved)
  / 40 secondary test (batch 3, generated after the paper's model development).
* Target: log10(cycle life). Standardised features, elastic net with alpha
  and l1_ratio chosen by repeated 4-fold CV on the training cells only.
* Metrics in cycles: RMSE and mean absolute percent error. As in the paper,
  primary test is also reported without cell b2c1 (148 cycles, the outlier).

Usage:
    python evaluation/build_cycle_life_features.py   # needs qdlin_first100.npz
    python evaluation/evaluate_cycle_life.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.cycle_life_features import ALL_FEATURES, PAPER_MODELS  # noqa: E402

FEATURES = BASE_DIR / "data" / "severson" / "processed" / "features.csv"
OUT_DIR = BASE_DIR / "evaluation"
OUTLIER_CELL = "b2c1"

# Severson et al. 2019, Table 2. Primary test values in brackets exclude the outlier cell.
PAPER_RESULTS = {
    "variance": {"rmse": {"train": 103, "test": 138, "test_no_outlier": 138, "secondary_test": 196},
                 "pct": {"train": 14.1, "test": 14.7, "test_no_outlier": 13.2, "secondary_test": 11.4}},
    "discharge": {"rmse": {"train": 76, "test": 91, "test_no_outlier": 86, "secondary_test": 173},
                  "pct": {"train": 9.8, "test": 13.0, "test_no_outlier": 10.1, "secondary_test": 8.6}},
    "full": {"rmse": {"train": 51, "test": 118, "test_no_outlier": 100, "secondary_test": 214},
             "pct": {"train": 5.6, "test": 14.1, "test_no_outlier": 7.5, "secondary_test": 10.7}},
}
EVAL_SETS = ["train", "test", "test_no_outlier", "secondary_test"]


def make_elastic_net():
    cv = RepeatedKFold(n_splits=4, n_repeats=10, random_state=0)
    return make_pipeline(StandardScaler(), ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=cv, max_iter=100_000))


def make_random_forest():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test = df[df.split == "test"]
    return {"train": df[df.split == "train"], "test": test,
            "test_no_outlier": test[test.cell_id != OUTLIER_CELL],
            "secondary_test": df[df.split == "secondary_test"]}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 1),
            "pct": round(float(np.mean(np.abs(y - p) / y) * 100), 1)}


def evaluate(df: pd.DataFrame, features: list[str] | None, make_model) -> tuple[dict, pd.Series]:
    """Fit on train cells, return metrics per evaluation set and predictions per cell."""
    sets = subsets(df)
    train = sets["train"]
    y_log = np.log10(train["cycle_life"].to_numpy())
    if features is None:  # baseline: every cell gets the training mean of log10(cycle life)
        pred = pd.Series(10 ** y_log.mean(), index=df.cell_id)
    else:
        model = make_model().fit(train[features], y_log)
        pred = pd.Series(10 ** model.predict(df[features]), index=df.cell_id)
    res = {name: metrics(s["cycle_life"].to_numpy(), pred[s.cell_id].to_numpy()) for name, s in sets.items()}
    # one number comparable to the abstract's 9.1%: all test cells except the outlier
    both = pd.concat([sets["test_no_outlier"], sets["secondary_test"]])
    res["all_test_no_outlier"] = metrics(both["cycle_life"].to_numpy(), pred[both.cell_id].to_numpy())
    return res, pred


MODELS = {
    "Baseline: training mean": (None, None),
    "Variance model (paper features)": (PAPER_MODELS["variance"], make_elastic_net),
    "Discharge model (paper features)": (PAPER_MODELS["discharge"], make_elastic_net),
    "Full model (paper features)": (PAPER_MODELS["full"], make_elastic_net),
    "Elastic net, all 20 candidate features": (ALL_FEATURES, make_elastic_net),
    "Random forest, all 20 candidate features": (ALL_FEATURES, make_random_forest),
}
PAPER_KEY = {"Variance model (paper features)": "variance", "Discharge model (paper features)": "discharge",
             "Full model (paper features)": "full"}


def main() -> None:
    df = pd.read_csv(FEATURES)
    results = {"dataset": {"cells": len(df), **df.split.value_counts().to_dict()},
               "models": {}, "paper": PAPER_RESULTS}
    preds = {}
    for name, (feats, make_model) in MODELS.items():
        res, pred = evaluate(df, feats, make_model)
        results["models"][name] = res
        preds[name] = pred.round(0)
        print(name, {k: v for k, v in res.items()})

    per_cell = df.set_index("cell_id")[["split", "cycle_life"]].join(pd.DataFrame(preds))
    per_cell.to_csv(OUT_DIR / "cycle_life_predictions.csv")
    (OUT_DIR / "cycle_life_results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results)
    print("wrote evaluation/cycle_life_results.md, .json and cycle_life_predictions.csv")


def write_markdown(r: dict) -> None:
    ds = r["dataset"]
    head = "| Model | Train | Primary test | Primary test w/o outlier | Secondary test |"
    lines = [
        "# Early cycle life prediction: Severson et al. (2019)",
        "",
        "Generated by `python evaluation/evaluate_cycle_life.py`. Task: predict total cycle life "
        "(cycles to 80% of nominal capacity) from the first 100 cycles.",
        f"Data: {ds['cells']} LFP/graphite cells ({ds['train']} train / {ds['test']} primary test / "
        f"{ds['secondary_test']} secondary test, paper split). Not combined with the NASA data.",
        "",
        "## Mean absolute percent error",
        "",
        head, "|---|---|---|---|---|",
    ]

    def row(label, values):
        return f"| {label} | " + " | ".join(str(values[s]) for s in EVAL_SETS) + " |"

    for name, res in r["models"].items():
        lines.append(row(name, {s: res[s]["pct"] for s in EVAL_SETS}))
        if name in PAPER_KEY:
            lines.append(row(f"↳ paper, {PAPER_KEY[name]} model", PAPER_RESULTS[PAPER_KEY[name]]["pct"]))
    lines += ["", "## RMSE (cycles)", "", head, "|---|---|---|---|---|"]
    for name, res in r["models"].items():
        lines.append(row(name, {s: res[s]["rmse"] for s in EVAL_SETS}))
        if name in PAPER_KEY:
            lines.append(row(f"↳ paper, {PAPER_KEY[name]} model", PAPER_RESULTS[PAPER_KEY[name]]["rmse"]))
    lines += ["", "## All test cells except the outlier (82 cells)", "",
              "The paper's abstract reports 9.1% test error. That is consistent with its full model averaged over "
              "the primary test without the outlier (7.5%, 42 cells) and the secondary test (10.7%, 40 cells): "
              "(7.5 x 42 + 10.7 x 40) / 82 = 9.1. The paper's full model also left out four cells whose "
              "temperature probe lost contact; they are kept here.",
              "", "| Model | Mean % error | RMSE (cycles) |", "|---|---|---|",
              "| Paper, full model (headline 9.1%) | 9.1 | - |"]
    for name, res in r["models"].items():
        lines.append(f"| {name} | {res['all_test_no_outlier']['pct']} | {res['all_test_no_outlier']['rmse']} |")
    (OUT_DIR / "cycle_life_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
