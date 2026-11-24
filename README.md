# PSLD: Port State Latent Dynamics

Code for **"A World Modeling Forecasting Framework for Maritime Port Operations Management"**: an observation-driven latent-dynamics model that forecasts the operational state of seaports from
AIS data. It is trained on eleven North American East Coast port complexes (hourly, 72 h ahead)
and applied to three unseen West Coast complexes.

This repository contains only what is needed to reproduce the paper's data, figure and model
results:

1. the data pipeline: downloading AIS pings and turning them into model-ready tensors;
2. training and evaluation of every model in the paper (one run each): PSLD, Graph WaveNet,
   DCRNN, VAR, SARIMA, seasonal-naive, and the West Coast transfer study.

## Layout

```
configs/ports.csv        the 16 port reference points (11 East/Gulf complexes, 3 West, Sault Ste Marie)
pipeline/                steps 00-07: AIS download -> preprocessing -> regions -> tensors
models/                  train / evaluate / score scripts
psld/                    shared code: paths, config, data loading, protocol and metrics,
                         the PSLD model (model.py), Graph WaveNet and DCRNN (stgnn.py)
requirements.txt
```

Everything the code reads or writes lives under one data root, `data/` by default (set the
`PSLD_DATA` environment variable to put it elsewhere, e.g. on an external disk):

```
data/staging  staged_pings  derived  preprocessed  geo  graph  work  tensors  results  figures
```

Run every script from the repository root.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

* Python 3.10 or newer; developed on 3.14.
* PyTorch picks CUDA, then Apple MPS, then CPU. The paper's runs used an Apple M4 Pro (14 cores,
  24 GB, MPS);
* DuckDB installs its `h3` extension from the community repository on first use, so the pipeline
  needs internet access.
* Disk: the AIS archive is about 41 GB compressed, but raw files are streamed (downloaded,
  processed, deleted), so peak use during download is small. The preprocessed pings are about
  13 GB, the staged per-day subsets about 22 GB before step 2.

## 1. Data pipeline

AIS broadcast points come from the public U.S. Marine Cadastre / NOAA archive
(`https://noaaocm.blob.core.windows.net/ais/csv2/`), 2023-01-01 to 2026-06-30. Geometry uses
Natural Earth 10 m land, coastline and lakes (public domain).

| Step | Script | What it does | Writes |
|---|---|---|---|
| 0 | `pipeline/00_download_geo.py` | download Natural Earth shapefiles | `geo/` |
| 1 | `pipeline/01_fetch_ais.py <year> [n_parallel]` | stream each daily file; keep Class A pings in the study domain, index them to H3 resolution 10, flag stationary pings; per-vessel and per-hexagon daily summaries | `staged_pings/`, `derived/` |
| 2 | `pipeline/02_consolidate_pings.py` | resolve vessel attributes once over the archive, apply the final fleet filter (type 70-89, length >= 100 m), remove isolated position jumps (implied speed > 60 kn on both sides) | `preprocessed/` |
| 3 | `pipeline/03_build_regions.py` | fit and freeze the node geometry on the training period less seven days: berth, anchorage (destination-aware rule) and approach regions per port | `graph/` |
| 4 | `pipeline/04_label_pings.py` | assign every ping to a node; hourly occupancy, 30-minute stationary buckets, approach-ring crossings | `work/` |
| 5 | `pipeline/05_build_spells.py` | vessel stays (spells) and the duration event table | `work/` |
| 6 | `pipeline/06_build_edges.py` | intra-port and inter-port (origin-destination) edges, measured on the training period only | `graph/` |
| 7 | `pipeline/07_build_panel.py` | hourly node features, the four targets, masks (including reception outages), calendar forcing, scalers, splits | `tensors/` |

```bash
python pipeline/00_download_geo.py
for y in 2023 2024 2025 2026; do python pipeline/01_fetch_ais.py $y; done
python pipeline/02_consolidate_pings.py
python pipeline/03_build_regions.py
python pipeline/04_label_pings.py
python pipeline/05_build_spells.py
python pipeline/06_build_edges.py
python pipeline/07_build_panel.py                 # East + Gulf graph (the training region)
SCOPE=all python pipeline/07_build_panel.py       # adds the West Coast nodes for the transfer study
```

`SCOPE` selects which nodes the tensors contain: `graph` (default) is the 33 East and Gulf nodes;
`all` adds the held-out West Coast complexes (and Sault Ste Marie, which is excluded from the
study). Tensors for `all` carry an `_all` suffix.

Splits are chronological: training to 2025-12-31, validation 2026-01-01 to 2026-03-31, test
2026-04-01 to 2026-06-30. All scalers are fitted on the training period only.

## 2. Models

All models see the same tensors, splits and rolling test origins (every 24 h through the test
quarter, 72 h horizon), and are scored on real vessel counts and hours.

```bash
# PSLD (graph-free model reported in the paper)
python models/train_psld.py                       # -> results/checkpoints/psld.pt, results/psld_quantiles.npy

# Graph baselines: point head (Table I) and nine-quantile head (Table II)
python models/train_stgnn.py gwnet
python models/train_stgnn.py dcrnn
python models/train_stgnn.py gwnet --head quantile
python models/train_stgnn.py dcrnn --head quantile

# Classical baselines: seasonal-naive, climatology, VAR, SARIMA
python models/run_baselines.py

# Tables I and II
python models/score_east.py                       # -> results/table_mae.csv, table_crps.csv
```

West Coast transfer: PSLD is applied to the unseen complexes with the East Coast
weights unchanged (`zeroshot`), and optionally fine-tuned on West Coast training data
(`finetune`); the classical and graph baselines are fitted directly on West Coast data:

```bash
export SCOPE=all
python models/run_baselines.py --domain west
python models/train_stgnn.py gwnet --domain west
python models/train_stgnn.py dcrnn --domain west
python models/west_transfer.py zeroshot
python models/west_transfer.py finetune
python models/score_west.py                       # -> results/table_west_mae.csv
```

Each training script takes `--seed` for further runs and `--eval-only --ckpt <file>` to forecast
from a saved checkpoint; `train_psld.py --intra-port-graph` adds the intra-port message-passing
step. Training recipes (context length, optimiser, schedule, early stopping) are documented in
the docstring at the top of each script.


## Data availability

The AIS archive is publicly available from the U.S. Marine Cadastre. No data derived from it is
distributed with this repository; the pipeline regenerates all of it.
