"""Loading, scaling and windowing for the tensors built by pipeline/07_build_panel.py.

Every scaler is fitted on TRAIN only (tensors/scalers.json) and applied unchanged
to val and test; predictions are inverted before any metric, so all models are
compared on real vessel counts and hours.
"""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from psld import paths, config

TAG = config.TAG

# static node descriptors (time-invariant); everything else in X is a dynamic feature
STATIC_FEATURES = ["is_berth", "is_anchorage", "is_approach", "n_hexes", "area_km2",
                   "train_mean_occupancy", "approach_radius_km", "draft_era"]
INTRA_PORT = ("A2Q", "Q2B", "A2B", "B2A")      # approach->queue->berth structural edges


def load():
    t = paths.TENSORS
    d = dict(X=np.load(t / f"X{TAG}.npy"), Y=np.load(t / f"Y{TAG}.npy"),
             U=np.load(t / f"U{TAG}.npy"), mask=np.load(t / f"mask{TAG}.npy"),
             target_mask=np.load(t / f"target_mask{TAG}.npy"),
             scalers=json.load(open(t / f"scalers{TAG}.json")),
             split=json.load(open(t / f"split_index{TAG}.json")),
             targets=json.load(open(t / f"target_index{TAG}.json")),
             hours=pd.read_parquet(t / f"hours{TAG}.parquet").hour)
    order = json.load(open(paths.GRAPH / "node_order.json"))["node_order"]
    if config.SCOPE == "all":
        nd = pd.read_parquet(paths.GRAPH / "nodes.parquet")
        order = order + [n for n in nd.node_id if n not in order]
    d["nodes"] = order
    d["feats"] = d["scalers"]["features"]
    return d


def feature_split(d):
    """Indices of the dynamic and the static columns of X."""
    feats = d["feats"]
    dyn = [i for i, f in enumerate(feats) if f not in STATIC_FEATURES]
    sta = [feats.index(f) for f in STATIC_FEATURES]
    return dyn, sta


def scale(X, scalers, invert=False):
    """Apply (or undo) the train-fitted per-feature transform. X is [..., F]."""
    out = X.astype(np.float64).copy()
    for i, f in enumerate(scalers["features"]):
        p = scalers["per_feature"][f]
        if p["transform"] == "none":
            continue
        if p["transform"] == "log1p+standardize":
            out[..., i] = (np.expm1(out[..., i] * p["std"] + p["mean"]) if invert
                           else (np.log1p(out[..., i]) - p["mean"]) / p["std"])
        else:                                    # robust
            out[..., i] = (out[..., i] * p["iqr"] + p["median"] if invert
                           else (out[..., i] - p["median"]) / p["iqr"])
    return out.astype(np.float32)


def prepare_targets(d):
    """Standardised targets and the 'anchor' (last observed target value).

    Targets are log1p-transformed, then standardised per (node, channel) by the
    train-period median and inter-quartile range, using observed cells only
    (log1p because all four channels are non-negative and right-skewed; robust
    because a handful of 300-hour waits would otherwise set the scale of a node).

    Returns Ys [T,N,4] (standardised), med / iqr [N,4] (to invert), and Yanchor
    [T,N,4]: the most recent observed standardised value at or before each hour
    (unobserved hours are forward-filled, so the anchor never takes a value the
    channel did not take).
    """
    Y, TM = d["Y"], d["target_mask"]
    T, N, _ = Y.shape
    tr0, tr1 = d["split"]["train"]
    Ylog = np.log1p(np.clip(Y, 0, None))
    med = np.zeros((N, 4), np.float32)
    iqr = np.ones((N, 4), np.float32)
    for i in range(N):
        for c in range(4):
            v = Ylog[tr0:tr1, i, c][TM[tr0:tr1, i, c]]
            if len(v) > 10:
                q1, q3 = np.percentile(v, [25, 75])
                med[i, c] = np.median(v)
                iqr[i, c] = max(q3 - q1, 1e-3)
    Ys = ((Ylog - med) / iqr).astype(np.float32)
    anchor = pd.DataFrame(np.where(TM, Ys, np.nan).reshape(T, -1)).ffill().bfill()
    anchor = anchor.values.reshape(T, N, 4).astype(np.float32)
    return Ys, med, iqr, anchor


def invert_targets(Z, med, iqr):
    """Standardised log values back to real units (vessels or hours)."""
    return np.expm1(np.clip(Z * iqr + med, 0, 20))


def adjacency(d, relations=None):
    """Weighted directed adjacency A[src, dst] over the panel's nodes.

    Edge weight is log(voyage count) * observation weight, summed over the chosen
    relations (all measured relations by default; INTRA_PORT for the intra-port
    graph).  Edges whose endpoints are not in the panel are ignored.
    """
    edges = pd.read_parquet(paths.GRAPH / "edges.parquet")
    if relations is not None:
        edges = edges[edges.relation.isin(relations)]
    nidx = {n: i for i, n in enumerate(d["nodes"])}
    N = len(nidx)
    A = np.zeros((N, N))
    for e in edges.itertuples():
        if e.src_node in nidx and e.dst_node in nidx:
            A[nidx[e.src_node], nidx[e.dst_node]] += e.log_count * e.obs_weight
    return A


def west_nodes(d):
    """Node indices of the held-out West-coast complexes (Sault Ste Marie, a lock
    queue rather than a port call, is excluded).  Requires SCOPE=all."""
    nodes = pd.read_parquet(paths.GRAPH / "nodes.parquet").set_index("node_id")
    return [i for i, n in enumerate(d["nodes"])
            if not bool(nodes.loc[n, "in_graph"]) and nodes.loc[n, "port"] != "Sault Ste Marie"]
