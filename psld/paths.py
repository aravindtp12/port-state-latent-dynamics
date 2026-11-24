"""Where every file lives.

Everything the code reads or writes (AIS pings, graph, tensors, checkpoints,
predictions, figures) sits under one data root, which is git-ignored.  It defaults
to ``<repo>/data`` and can be moved with the ``PSLD_DATA`` environment variable,
e.g. onto an external disk -- the preprocessed pings alone are ~13 GB.

    data/
      staging/          raw daily CSVs while they are being processed; DuckDB spill
      staged_pings/     loose per-day ping subsets (step 1)
      derived/          per-vessel-day and H3-res-7 daily summaries (step 1)
      preprocessed/     final filtered pings + vessel table (step 2)
      geo/              Natural Earth shapefiles (downloaded by pipeline/00_download_geo.py)
      graph/            node geometry, edges, adjacency matrices (steps 3, 6)
      work/             per-(node, hour) tables built from the pings (step 4)
      tensors/          model-ready arrays X, Y, U, masks, scalers (step 7)
      results/          predictions and checkpoints written by models/*.py
      figures/          figures written by figures/*.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PSLD_DATA", ROOT / "data")).expanduser()
CONFIGS = ROOT / "configs"

STAGING = DATA / "staging"
STAGED = DATA / "staged_pings"
DERIVED = DATA / "derived"
PREPROCESSED = DATA / "preprocessed"
PINGS = PREPROCESSED / "pings"
VESSELS = PREPROCESSED / "vessels.parquet"
GEO = DATA / "geo"
GRAPH = DATA / "graph"
WORK = DATA / "work"
TENSORS = DATA / "tensors"
RESULTS = DATA / "results"
CKPT = RESULTS / "checkpoints"
FIGURES = DATA / "figures"

PORTS_CSV = CONFIGS / "ports.csv"
DUCKDB_TMP = STAGING / "duckdb_tmp"


def ensure(*dirs):
    """Create directories (and parents) if missing; returns the last one."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return Path(dirs[-1])
