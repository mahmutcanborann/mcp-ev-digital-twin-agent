# Severson et al. (2019) processed data

Built by `evaluation/build_severson_dataset.py` from the three raw `.mat` batch
files in `data/severson/raw/` (source: https://data.matr.io/1/, Severson, Attia
et al., *Nature Energy* 2019).

This is a commercial LFP/graphite (A123 APR18650M1A) fast-charging dataset. It
is **never merged with the NASA data**: different chemistry, test protocol,
model and evaluation. Task: predict total cycle life from the first 100 cycles.

## Files

| File | Content |
|---|---|
| `cells.csv` | One row per cell: `cell_id` (e.g. `b1c0`), `batch` (1-3), `policy` (charging protocol), `cycle_life` (cycles to 80% of nominal capacity), `split` |
| `summary.csv.gz` | One row per cell and cycle: `cycle`, `q_discharge`, `q_charge` (Ah), `ir` (internal resistance, Ohm), `t_max`, `t_avg`, `t_min` (°C), `charge_time` (min), `cell_id` |
| `features.csv` | One row per cell: split, `cycle_life` and the 20 early-cycle features (paper Table 1), built by `evaluation/build_cycle_life_features.py`; see `src/cycle_life_features.py` |
| `qdlin_first100.npz` | One array per `cell_id`, shape (101, 1000): discharge capacity Qd(V) interpolated on a fixed 1000-point voltage grid. Row j is `cycles[j]`, matching the authors' indexing; batch 1 cycle 0 has no curve and is NaN |

`cells.csv`, `summary.csv.gz` and `features.csv` are in git. `qdlin_first100.npz` (~42 MB) is
not; to regenerate it, download the three `.mat` files from
https://data.matr.io/1/ into `data/severson/raw/` and run:

```bash
python evaluation/build_severson_dataset.py --raw data/severson/raw --out data/severson/processed
```

## Cleaning (follows the authors' LoadData script)

- Batch 1: drop cells that did not reach 80% capacity (c8, c10, c12, c13, c22).
- Batch 2: cells c7, c8, c9, c15, c16 continue batch 1 cells c0..c4. Their
  summaries are appended and their cycle life minus 1 is added to the batch 1
  cell (LoadData `add_len`); the batch 2 entries are then dropped.
- Batch 3: drop noisy channels (c2, c23, c32, c37, c42, c43).
- Result: 41 + 43 + 40 = 124 cells, cycle life 148 to 2237.

## Split (as in the paper)

Batches 1 and 2 are interleaved: every other cell goes to `test` (plus the last
cell), the rest to `train`. Batch 3 is `secondary_test`.

| split | cells |
|---|---|
| train | 41 |
| test | 43 |
| secondary_test | 40 |
