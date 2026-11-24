"""Study period, chronological splits and panel scope.

SCOPE (environment variable) selects which nodes the model tensors contain:
  graph  the 33 East + Gulf nodes (11 port complexes x {approach, anchorage, berth});
         this is the training region and the graph the models see.  Default.
  all    the East+Gulf nodes plus the held-out West-coast complexes (Long Beach,
         Oakland, Vancouver) and Sault Ste Marie, used only for the transfer study.
Tensors for SCOPE=all are written/read with an ``_all`` suffix.
"""
import os
import pandas as pd

SCOPE = os.environ.get("SCOPE", "graph")
assert SCOPE in ("graph", "all"), "SCOPE must be 'graph' or 'all'"

SPAN0, SPAN1 = "2023-01-01", "2026-06-30"
# node geometry is fitted on the training period less a 7-day buffer (the
# anchorage rule looks 7 days forward), then frozen for validation and test
FIT_END, TRAIN_END, VAL_END, TEST_END = "2025-12-24", "2025-12-31", "2026-03-31", "2026-06-30"

_day = pd.Timedelta("1D")
SPLITS = {
    "train": (SPAN0, TRAIN_END),
    "val": (str((pd.Timestamp(TRAIN_END) + _day).date()), VAL_END),
    "test": (str((pd.Timestamp(VAL_END) + _day).date()), TEST_END),
}
TAG = "" if SCOPE == "graph" else f"_{SCOPE}"


def banner():
    print(f"[config] scope={SCOPE} | geometry fit ends {FIT_END} | "
          f"train..{TRAIN_END} val..{VAL_END} test..{TEST_END}")
