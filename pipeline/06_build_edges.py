"""Edge tensors, measured on the training period only.

Family 1 - intra-port structure, the same four directed edges at every port:
    A_p -> Q_p   an inbound ship joins the queue
    Q_p -> B_p   a queued ship takes a berth
    A_p -> B_p   an inbound ship berths without queueing
    B_p -> A_p   a ship departs outward across the ring
Family 2 - inter-port voyages B_p -> B_q, from consecutive berth spells of the
same vessel at different complexes.

Half the directed port pairs are never observed.  That means UNSEEN, not zero
flow, so every edge carries obs_weight = n / (n + PRIOR): a pair seen twice is
down-weighted rather than trusted, and an unseen pair simply has no edge.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from psld import paths, config as cfg
paths.ensure(paths.DUCKDB_TMP, paths.GRAPH, paths.WORK, paths.TENSORS, paths.RESULTS)
import numpy as np, pandas as pd, duckdb, scipy.sparse as sp, json
pd.set_option("display.width", 250, "display.max_columns", 60)
FIT0, FIT1 = cfg.SPAN0, cfg.TRAIN_END        # training period
QUEUE_H, APPROACH_D, MAX_TRANSIT_D, PRIOR = 24, 7, 21, 5.0

nodes = pd.read_parquet(f"{paths.GRAPH}/nodes.parquet")
ing = nodes[nodes.in_graph].reset_index(drop=True)
idx = {n: i for i, n in enumerate(ing.node_id)}
N = len(ing)
kind_of = dict(zip(nodes.node_id, nodes.node_kind))
cx_of = dict(zip(nodes.node_id, nodes["complex"]))
port_node = {(r["complex"], r.node_kind): r.node_id for _, r in nodes.iterrows()}

sp_all = pd.read_parquet(paths.WORK / "spells.parquet")
sp_all = sp_all[(sp_all.t_in >= FIT0) & (sp_all.t_in <= f"{FIT1} 23:59:59")]
cross = duckdb.sql(f"""SELECT * FROM '{paths.WORK}/crossings/*.parquet'
    WHERE t_cross BETWEEN TIMESTAMP '{FIT0}' AND TIMESTAMP '{FIT1} 23:59:59'""").df()
cross["complex"] = cross.node_id.map(cx_of)
print(f"training window {FIT0} .. {FIT1}: {len(sp_all):,} spells, {len(cross):,} ring crossings")

B = sp_all[sp_all.node_kind == "berth"][["mmsi","complex","port","t_in","t_out"]]
Q = sp_all[sp_all.node_kind == "anchorage"][["mmsi","complex","t_in","t_out"]]
cin = cross[cross.direction == "in"][["mmsi","complex","t_cross"]]
cout = cross[cross.direction == "out"][["mmsi","complex","t_cross"]]

def handover(src, dst, src_t, dst_t, within):
    """Rows of dst matched to the latest preceding row of src at the same port."""
    m = src.merge(dst, on=["mmsi","complex"], suffixes=("_s","_d"))
    dt = m[dst_t] - m[src_t]
    m = m[(dt >= pd.Timedelta(0)) & (dt <= within)]
    if not len(m): return 0, np.nan, m
    g = m.groupby(["mmsi","complex",dst_t], as_index=False)[src_t].max()
    return len(g), float(((g[dst_t]-g[src_t]).dt.total_seconds()/3600).median()), g

print("\n=== family 1: intra-port structure ===")
rows = []
# A -> Q : ring crossing then an anchorage spell
n_aq, h_aq, m_aq = handover(cin.rename(columns={"t_cross":"t_s"}),
                            Q.rename(columns={"t_in":"t_d"})[["mmsi","complex","t_d"]],
                            "t_s", "t_d", pd.Timedelta(days=APPROACH_D))
# Q -> B : anchorage spell then a berth spell at the same port
n_qb, h_qb, m_qb = handover(Q.rename(columns={"t_out":"t_s"})[["mmsi","complex","t_s"]],
                            B.rename(columns={"t_in":"t_d"})[["mmsi","complex","t_d"]],
                            "t_s", "t_d", pd.Timedelta(hours=QUEUE_H))
# A -> B : ring crossing then a berth call with no anchorage stay in between
n_ab, h_ab, m_ab = handover(cin.rename(columns={"t_cross":"t_s"}),
                            B.rename(columns={"t_in":"t_d"})[["mmsi","complex","t_d"]],
                            "t_s", "t_d", pd.Timedelta(days=APPROACH_D))
queued = set(map(tuple, m_qb[["mmsi","complex","t_d"]].values))
direct = m_ab[[tuple(r) not in queued for r in m_ab[["mmsi","complex","t_d"]].values]]
n_ab, h_ab = len(direct), float(((direct.t_d-direct.t_s).dt.total_seconds()/3600).median())
# B -> A : berth spell then an outbound ring crossing
n_ba, h_ba, _ = handover(B.rename(columns={"t_out":"t_s"})[["mmsi","complex","t_s"]],
                         cout.rename(columns={"t_cross":"t_d"}),
                         "t_s", "t_d", pd.Timedelta(days=APPROACH_D))

per = []
for cx in sorted(ing["complex"].unique()):
    for rel, m, src, dst in [("A2Q", m_aq, "approach", "anchorage"),
                             ("Q2B", m_qb, "anchorage", "berth"),
                             ("A2B", direct, "approach", "berth"),
                             ("B2A", None, "berth", "approach")]:
        if rel == "B2A":
            sub = handover(B[B["complex"]==cx].rename(columns={"t_out":"t_s"})[["mmsi","complex","t_s"]],
                           cout[cout["complex"]==cx].rename(columns={"t_cross":"t_d"}),
                           "t_s", "t_d", pd.Timedelta(days=APPROACH_D))
            n, h = sub[0], sub[1]
        else:
            g = m[m["complex"] == cx]
            n = len(g)
            h = float(((g.t_d-g.t_s).dt.total_seconds()/3600).median()) if n else np.nan
        per.append(dict(src_node=port_node[(cx, src)], dst_node=port_node[(cx, dst)],
                        relation=rel, port=ing[ing["complex"]==cx].port.iloc[0],
                        count=n, median_transit_h=round(h, 2) if n else np.nan))
fam1 = pd.DataFrame(per)
print(fam1.groupby("relation").agg(edges=("count","size"), transitions=("count","sum"),
      median_handover_h=("median_transit_h","median")).to_string())
pv = fam1.pivot(index="port", columns="relation", values="count")
pv["pct_queued"] = (100*pv.Q2B/(pv.Q2B+pv.A2B)).round(1)
print("\nper port (transitions in the training window):")
print(pv.sort_values("Q2B", ascending=False).to_string())
print(f"\noverall {100*pv.Q2B.sum()/(pv.Q2B.sum()+pv.A2B.sum()):.1f}% of berth calls are"
      f" preceded by an anchorage stay at the same port within {QUEUE_H} h")

print("\n=== family 2: inter-port voyages ===")
sb = B.sort_values(["mmsi","t_in"]).copy()
sb["next_complex"] = sb.groupby("mmsi")["complex"].shift(-1)
sb["next_t_in"] = sb.groupby("mmsi").t_in.shift(-1)
od = sb[sb.next_complex.notna() & (sb["complex"] != sb.next_complex)].copy()
od["transit_h"] = (od.next_t_in - od.t_out).dt.total_seconds()/3600
od = od[od.transit_h.between(0, 24*MAX_TRANSIT_D)]
od["next_complex"] = od.next_complex.astype(int)
mat = (od.groupby(["complex","next_complex"])
         .agg(voyage_count=("mmsi","size"), vessels=("mmsi","nunique"),
              median_transit_h=("transit_h","median")).reset_index())
mat["src_node"] = [port_node[(c,"berth")] for c in mat["complex"]]
mat["dst_node"] = [port_node[(c,"berth")] for c in mat.next_complex]
mat["src_region"] = mat["complex"].map(dict(zip(nodes["complex"], nodes.region)))
mat["dst_region"] = mat.next_complex.map(dict(zip(nodes["complex"], nodes.region)))
cross_coast = mat[((mat.src_region.isin(["E","G"])) & (mat.dst_region == "W")) |
                  ((mat.src_region == "W") & (mat.dst_region.isin(["E","G"])))]
print(f"  {len(od):,} port-to-port voyages, {od.mmsi.nunique():,} vessels")
print(f"  cross-coast (East/Gulf <-> West): {int(cross_coast.voyage_count.sum())} voyages"
      f" over {len(cross_coast)} pairs -- excluded from the graph")
ing_cx = set(ing["complex"])
mat = mat[mat["complex"].isin(ing_cx) & mat.next_complex.isin(ing_cx)].copy()
npairs = len(ing_cx)*(len(ing_cx)-1)
print(f"  within East+Gulf: {len(mat)} of {npairs} directed pairs populated "
      f"({100*len(mat)/npairs:.0f}%), {int(mat.voyage_count.sum()):,} voyages")
print(f"  pairs with >=5 voyages: {int((mat.voyage_count>=5).sum())};"
      f"  >=50: {int((mat.voyage_count>=50).sum())}")
mat["relation"] = "OD"
mat["port"] = mat["complex"].map(dict(zip(nodes["complex"], nodes.port)))
top = mat.nlargest(10, "voyage_count")[["src_node","dst_node","voyage_count","median_transit_h"]]
print("\ntop 10 origin-destination edges:")
print(top.to_string(index=False))

print("\n=== write graph/edges.parquet and the adjacency matrices ===")
fam1 = fam1.rename(columns={"count":"voyage_count"})
edges = pd.concat([fam1[["src_node","dst_node","relation","voyage_count","median_transit_h"]],
                   mat[["src_node","dst_node","relation","voyage_count","median_transit_h"]]],
                  ignore_index=True)
edges["obs_weight"] = (edges.voyage_count/(edges.voyage_count+PRIOR)).round(4)
edges["log_count"] = np.log1p(edges.voyage_count).round(4)
edges = edges[edges.voyage_count > 0].reset_index(drop=True)
edges.to_parquet(f"{paths.GRAPH}/edges.parquet", index=False)
for rel, g in edges.groupby("relation"):
    r = [idx[s] for s in g.src_node]; c = [idx[d] for d in g.dst_node]
    A = sp.coo_matrix((g.log_count*g.obs_weight, (r, c)), shape=(N, N)).tocsr()
    sp.save_npz(f"{paths.GRAPH}/adj_{rel}.npz", A)
    print(f"  graph/adj_{rel}.npz  {A.nnz} edges, weight sum {A.sum():.1f}")
json.dump({"node_order": ing.node_id.tolist()}, open(f"{paths.GRAPH}/node_order.json","w"), indent=1)
print(f"  graph/edges.parquet: {len(edges)} edges over {N} nodes")
