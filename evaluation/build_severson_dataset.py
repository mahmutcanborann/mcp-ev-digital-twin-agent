"""Convert the Severson et al. (2019) .mat batch files into compact tables.

Source: https://data.matr.io/1/ (Severson, Attia et al., Nature Energy 2019).
This dataset is kept completely separate from the NASA data: different
chemistry (LFP/graphite), different test protocol, separate model and
separate evaluation. Task: early cycle life prediction (predict total
cycle life from the first 100 cycles).

Cleaning follows the authors' LoadData script:
- batch 1: drop cells that did not reach 80% capacity (c8, c10, c12, c13, c22)
- batch 2: cells c7, c8, c9, c15, c16 are continuations of batch 1 cells
  c0..c4; their cycles are appended to those cells and then dropped
- batch 3: drop noisy channels (c2, c23, c32, c37, c42, c43)
Result: 41 + 43 + 40 = 124 cells.
Batch 1 cycle 0 has no discharge curve (MATLAB empty); its Qdlin row is NaN. Row j of qdlin is cycles[j], matching the authors' indexing.

Outputs (in --out):
- cells.csv           one row per cell: id, batch, policy, cycle_life, split
- summary.csv.gz      one row per cell and cycle: capacities, IR, temperatures, charge time
- qdlin_first100.npz  interpolated discharge capacity Qd(V) for cycles 1..100 (for dQ features)

Usage:
    python build_severson_dataset.py --raw data/severson/raw --out data/severson/processed
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BATCH_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BATCH1_DROP = [8, 10, 12, 13, 22]
BATCH2_CONTINUATION = {7: 0, 8: 1, 9: 2, 15: 3, 16: 4}  # batch 2 cell -> batch 1 cell
BATCH3_DROP = [2, 23, 32, 37, 42, 43]
SUMMARY_FIELDS = {
    "cycle": "cycle", "QDischarge": "q_discharge", "QCharge": "q_charge", "IR": "ir",
    "Tmax": "t_max", "Tavg": "t_avg", "Tmin": "t_min", "chargetime": "charge_time",
}
N_EARLY = 100


def _read_text(f, ref):
    return f[ref][()].tobytes()[::2].decode()


def load_batch(path):
    cells = {}
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        for i in range(batch["summary"].shape[0]):
            cycle_life = float(np.ravel(f[batch["cycle_life"][i, 0]][()])[0])
            policy = _read_text(f, batch["policy_readable"][i, 0])
            summ = f[batch["summary"][i, 0]]
            summary = pd.DataFrame({out: np.ravel(summ[key][()]) for key, out in SUMMARY_FIELDS.items()})
            cyc = f[batch["cycles"][i, 0]]
            n_cycles = cyc["Qdlin"].shape[0]
            qdlin = np.full((N_EARLY + 1, 1000), np.nan)
            for j in range(min(n_cycles, N_EARLY + 1)):
                values = np.ravel(f[cyc["Qdlin"][j, 0]][()])
                if values.size == 1000:
                    qdlin[j] = values
            cells[i] = {"cycle_life": cycle_life, "policy": policy, "summary": summary, "qdlin": qdlin}
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    b1, b2, b3 = (load_batch(args.raw / BATCH_FILES[k]) for k in (1, 2, 3))

    for c in BATCH1_DROP:
        del b1[c]
    for c2, c1 in BATCH2_CONTINUATION.items():
        cont = b2.pop(c2)
        tail = cont["summary"].copy()
        tail["cycle"] += len(b1[c1]["summary"])
        # LoadData's add_len = [662, 981, 1060, 208, 482], i.e. continuation cycle_life - 1
        b1[c1]["cycle_life"] += cont["cycle_life"] - 1
        b1[c1]["summary"] = pd.concat([b1[c1]["summary"], tail], ignore_index=True)
    for c in BATCH3_DROP:
        del b3[c]

    ordered = [(f"b{k}c{i}", k, cell) for k, b in ((1, b1), (2, b2), (3, b3)) for i, cell in sorted(b.items())]
    assert len(ordered) == 124, len(ordered)

    # Paper split: batches 1+2 interleaved into train / primary test, batch 3 = secondary test.
    n12 = sum(1 for _, k, _ in ordered if k in (1, 2))
    test_idx = set(range(0, n12, 2)) | {n12 - 1}
    rows, summaries, qd = [], [], {}
    for idx, (cid, k, cell) in enumerate(ordered):
        split = "secondary_test" if k == 3 else ("test" if idx in test_idx else "train")
        rows.append({"cell_id": cid, "batch": k, "policy": cell["policy"],
                     "cycle_life": cell["cycle_life"], "split": split})
        s = cell["summary"].assign(cell_id=cid)
        summaries.append(s[s["cycle"] > 0])
        qd[cid] = cell["qdlin"].astype(np.float32)

    pd.DataFrame(rows).to_csv(args.out / "cells.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(args.out / "summary.csv.gz", index=False)
    np.savez_compressed(args.out / "qdlin_first100.npz", **qd)
    print(pd.DataFrame(rows).groupby("split")["cycle_life"].describe())


if __name__ == "__main__":
    main()
