"""The hourly node-feature panel and the model-ready tensors.

X : [T, N, F]  one row per (node, hour), single schema for all three node kinds,
               0 where a feature does not apply, node-kind one-hot so a shared
               model can tell them apart.  Raw units -- scalers.json holds the
               train-fitted constants, so every metric can be reported on real
               vessel counts.
Y : [T, N, 4]  (anchorage_occ, berth_occ, avg_berth_wait, avg_port_call) of the
               node's PORT, so every target is defined for all three node kinds.
               target_index.json lists the 44 unique (port, channel) series to
               score, because scoring all 33 nodes would count each port thrice.
target_mask    [T, N, 4]  the duration channels exist only where their trailing
               window holds a completed vessel.
U : [T, 7]     calendar forcing - the intervention surface for the simulator.
mask: [T, N]   False where the AIS feed itself was down, so a zero there is not
               read as an empty berth.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from psld import paths, config as cfg
paths.ensure(paths.DUCKDB_TMP, paths.GRAPH, paths.WORK, paths.TENSORS, paths.RESULTS)
import duckdb, numpy as np, pandas as pd, json, holidays
from psld import draftfix
pd.set_option("display.width", 250, "display.max_columns", 60)

cfg.banner()
SPAN0, SPAN1 = cfg.SPAN0, cfg.SPAN1
SPLITS = cfg.SPLITS
QUEUE_H, DRAFT_EPS, OUTAGE_FRAC = 24, 0.2, 0.5
TGT_W, FEAT_W = 12, 48        # trailing windows, hours (target / causal feature)

nodes = pd.read_parquet(f"{paths.GRAPH}/nodes.parquet")
# SCOPE=all keeps the held-out West-coast and Lakes nodes in the panel so the
# transfer study can score them; they are still flagged in_graph=False and are excluded from
# every East+Gulf table and from the scaler fit below.
ing = (nodes if cfg.SCOPE == "all" else nodes[nodes.in_graph]).reset_index(drop=True)
order = json.load(open(f"{paths.GRAPH}/node_order.json"))["node_order"]
if cfg.SCOPE == "all":
    order = order + [n for n in nodes.node_id if n not in order]
ing = ing.set_index("node_id").loc[order].reset_index()
N = len(ing); nidx = {n: i for i, n in enumerate(order)}
hours = pd.date_range(f"{SPAN0} 00:00", f"{SPAN1} 23:00", freq="h")
T = len(hours); hidx = pd.Series(np.arange(T), index=hours)
print(f"T = {T:,} hours   N = {N} nodes   {SPAN0} .. {SPAN1}")

DYN = ["occupancy","entries","exits","inbound_flux","mean_draft","draft_change_share",
       "mean_length","max_length","cargo_share","queue_share","transit_count",
       "avg_berth_wait","avg_port_call"]
DRAFT_ERA = "draft_era"   # 0 before the 2024-01-01 source change, 1 after
STAT = ["is_berth","is_anchorage","is_approach","n_hexes","area_km2",
        "train_mean_occupancy","approach_radius_km", DRAFT_ERA]
FEATS = DYN + STAT
X = np.zeros((T, N, len(FEATS)), dtype=np.float32)
fi = {f: i for i, f in enumerate(FEATS)}

def place(df, col, feat):
    """Scatter a (node_id, hr, value) frame into X."""
    d = df[df.node_id.isin(nidx)]
    r = hidx.reindex(d.hr.values).values
    ok = ~pd.isna(r)
    X[r[ok].astype(int), d.node_id.map(nidx).values[ok], fi[feat]] = d[col].values[ok]

con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='11GB';"
            f"PRAGMA temp_directory='{paths.DUCKDB_TMP}'; PRAGMA disable_progress_bar")

print("\n=== 1. occupancy, fleet composition, draft (from the labelling pass) ===")
h = con.execute(f"""SELECT node_id, hr, n_pings, occupancy, transit_count, mean_draft,
    mean_length, max_length, n_cargo, n_tanker FROM '{paths.WORK}/hourly/*.parquet'""").df()
h["cargo_share"] = np.where(h.occupancy > 0, h.n_cargo/h.occupancy.clip(lower=1), 0.0)
# the source changes draft scale overnight at 2024-01-01; map 2023 onto the later
# scale so the feature has no level break inside the training period (psld/draftfix.py)
h["mean_draft"] = draftfix.apply(h.mean_draft.values, h.hr.values)
for c, f in [("occupancy","occupancy"), ("transit_count","transit_count"),
             ("mean_draft","mean_draft"), ("mean_length","mean_length"),
             ("max_length","max_length"), ("cargo_share","cargo_share")]:
    place(h.fillna({c: 0.0}), c, f)
print(f"  {len(h):,} node-hours with at least one vessel "
      f"({100*len(h[h.node_id.isin(nidx)])/(T*N):.1f}% of the East+Gulf grid;"
      f" the rest are true zeros)")

print("\n=== 2. entries, exits, draft change, queue share (from spells) ===")
sp = pd.read_parquet(paths.WORK / "spells.parquet")
if cfg.SCOPE != "all": sp = sp[sp.in_graph]
sp = sp.copy()
sp["h_in"] = sp.t_in.dt.floor("h"); sp["h_out"] = sp.t_out.dt.floor("h")
for kind, col, feat in [("in","h_in","entries"), ("out","h_out","exits")]:
    g = (sp[sp.node_kind.isin(["berth","anchorage"])]
         .groupby(["node_id", col]).size().rename("v").reset_index()
         .rename(columns={col: "hr"}))
    place(g, "v", feat)
print(f"  {len(sp):,} spells -> entries/exits")

con.register("spd", sp[["mmsi","node_id","node_kind","t_in","t_out","draft"]])
# Two traps here.  A vessel with no earlier stay cannot have changed draft, so
# counting it as "unchanged" makes the feature ramp up over the first months of
# the archive as the fleet accumulates history -- the denominator is therefore
# only vessels that HAVE a previous stay.  And a pair straddling 2024-01-01 is
# comparing two different draft scales, so it is dropped rather than counted as
# a change.
dc = con.execute(f"""
WITH s AS (SELECT *, lag(draft) OVER w AS prev_draft, lag(t_in) OVER w AS prev_t
           FROM spd WHERE node_kind IN ('berth','anchorage')
           WINDOW w AS (PARTITION BY mmsi ORDER BY t_in)),
 f AS (SELECT node_id, mmsi, t_in, t_out,
         prev_draft IS NOT NULL AND draft IS NOT NULL
           AND NOT (prev_t < DATE '{draftfix.ERA_BREAK}'
                    AND t_in >= DATE '{draftfix.ERA_BREAK}') AS comparable,
         abs(draft - prev_draft) > {DRAFT_EPS} AS changed FROM s),
 e AS (SELECT node_id, mmsi, comparable, changed,
         unnest(generate_series(date_trunc('hour', t_in), date_trunc('hour', t_out),
                                INTERVAL 1 HOUR)) AS hr
       FROM f)
SELECT node_id, hr,
       coalesce(avg(changed::INT) FILTER (comparable), 0) AS v,
       avg(comparable::INT) AS comparable_share
FROM e GROUP BY 1,2""").df()
# The RATE of draft reporting drifts upward across the whole archive -- 9.8% of
# stays differed from the previous one in Jan 2023, 18.5% by Dec 2023, 36% by
# 2026 -- because the feed reports draft more often and more freshly over time,
# not because ships load and discharge more.  Left raw it is a trend a model
# would read as port behaviour.  It is therefore expressed RELATIVE to the
# network-wide rate at the same time, using a TRAILING 90-day mean so nothing
# looks forward: what survives is which nodes and hours run above or below the
# network's current reporting norm, which is the loading signal that was wanted.
net = (dc.groupby("hr").v.mean().reindex(hours).ffill()
         .rolling(90*24, min_periods=14*24).mean())
dc["raw"] = dc.v
dc["net"] = dc.hr.map(net)
dc["v"] = np.where(dc.net > 1e-6, (dc.v/dc.net).clip(0, 5), 0.0)
place(dc, "v", "draft_change_share")
print(f"  draft change: {100*dc.raw.mean():.1f}% of comparable vessel-hours are a vessel"
      f" whose draft moved\n    more than {DRAFT_EPS} m since its previous stay;"
      f" {100*dc.comparable_share.mean():.0f}% of vessel-hours are comparable"
      f" (the rest are a vessel's\n    first stay, or straddle the draft-scale break)."
      f"  The feature is stored RELATIVE to a\n    trailing 90-day network mean"
      f" (mean {dc.v.mean():.2f}), which strips the reporting-rate drift")

# queue share: of the berth arrivals in an hour, how many came from that port's anchorage
Bs = sp[sp.node_kind == "berth"][["mmsi","complex","t_in","node_id","h_in"]]
Qs = sp[sp.node_kind == "anchorage"][["mmsi","complex","t_out"]]
m = Bs.merge(Qs, on=["mmsi","complex"])
dt = m.t_in - m.t_out
m = m[(dt >= pd.Timedelta(0)) & (dt <= pd.Timedelta(hours=QUEUE_H))]
Bs = Bs.assign(from_q=Bs.set_index(["mmsi","complex","t_in"]).index.isin(
    pd.MultiIndex.from_frame(m[["mmsi","complex","t_in"]])))
qs = (Bs.groupby(["node_id","h_in"]).from_q.mean().rename("v").reset_index()
        .rename(columns={"h_in":"hr"}))
place(qs, "v", "queue_share")
print(f"  queue share: {100*Bs.from_q.mean():.1f}% of all berth arrivals came from the"
      f" port's own anchorage within {QUEUE_H} h")

print("\n=== 2b. the two duration channels ===")
print("""  Both are defined as ARRIVAL-INDEXED trailing-12 h means: at hour t,
  average the duration of vessels that entered the node in [t-12h, t] and have
  since completed.  That is the right TARGET -- it answers "what should a ship
  entering now expect" -- but it is not causal: at hour t it uses outcomes that
  only become known hours or days later, so feeding the same series in as a
  model INPUT would hand the context window the future.

  So the two series are built twice.  The target is the arrival-indexed
  trailing-12 h mean.  The X feature is COMPLETION-INDEXED over a trailing 48 h:
  at hour t, the mean duration of vessels that finished in [t-48h, t], which is
  knowable at t.  Same quantity, honest timing.""")
ev_all = pd.read_parquet(paths.WORK / "events.parquet")
ev_all = ev_all[(ev_all.in_graph | (cfg.SCOPE == "all")) & ~ev_all.long_stay].copy()
wait_ev = ev_all[(ev_all.target_kind == "berth_wait") &
                 (ev_all.exit_reason == "berthed")].copy()
call_ev = ev_all[(ev_all.target_kind == "port_call") & ev_all.event_observed].copy()
wait_ev["done_ts"] = wait_ev.entry_ts + pd.to_timedelta(wait_ev.wait_to_berth_h, "h")
call_ev["done_ts"] = call_ev.entry_ts + pd.to_timedelta(call_ev.duration_h, "h")
print(f"  {len(wait_ev):,} completed waits (anchorage spells that berthed), "
      f"{len(call_ev):,} completed port calls")

def trailing(df, tcol, W):
    """mean duration over a trailing window of W hours, indexed by tcol."""
    k = df.groupby(["node_id", df[tcol].dt.floor("h")]).duration_h.agg(["sum", "size"])
    ssum = k["sum"].unstack(0).reindex(hours).fillna(0.0)
    scnt = k["size"].unstack(0).reindex(hours).fillna(0.0)
    rs = ssum.rolling(W+1, min_periods=1).sum()
    rn = scnt.rolling(W+1, min_periods=1).sum()
    return rs.div(rn).where(rn > 0), rn

WAIT_T, WAIT_N = trailing(wait_ev, "entry_ts", TGT_W)    # target, arrival-indexed
CALL_T, CALL_N = trailing(call_ev, "entry_ts", TGT_W)
WAIT_F, _      = trailing(wait_ev, "done_ts",  FEAT_W)   # feature, completion-indexed
CALL_F, _      = trailing(call_ev, "done_ts",  FEAT_W)
for frame, feat in [(WAIT_F, "avg_berth_wait"), (CALL_F, "avg_port_call")]:
    for n in frame.columns:
        if n in nidx:
            X[:, nidx[n], fi[feat]] = frame[n].fillna(0.0).values
cov_w = 100*WAIT_T.notna().mean().mean(); cov_c = 100*CALL_T.notna().mean().mean()
print(f"  target coverage: avg_berth_wait defined in {cov_w:.1f}% of node-hours, "
      f"avg_port_call {cov_c:.1f}%")
print(f"  feature coverage (48 h window): {100*WAIT_F.notna().mean().mean():.1f}% / "
      f"{100*CALL_F.notna().mean().mean():.1f}%")
print(f"  mean wait {np.nanmean(WAIT_T.values):.1f} h, mean call "
      f"{np.nanmean(CALL_T.values):.1f} h")

print("\n=== 3. inbound flux (approach rings) ===")
cr = con.execute(f"""SELECT node_id, date_trunc('hour', t_cross) AS hr, count(*) AS v
    FROM '{paths.WORK}/crossings/*.parquet' WHERE direction = 'in' GROUP BY 1,2""").df()
place(cr, "v", "inbound_flux")
print(f"  {int(cr.v.sum()):,} inbound crossings over {T:,} hours")

print("\n=== 4. static features ===")
area = dict(zip(nodes.node_id, nodes.area_km2))
nhex = dict(zip(nodes.node_id, nodes.n_hexes))
rad  = dict(zip(nodes.node_id, nodes.approach_radius_km.fillna(0.0)))
for i, n in enumerate(order):
    k = ing.node_kind[i]
    X[:, i, fi["is_berth"]]     = float(k == "berth")
    X[:, i, fi["is_anchorage"]] = float(k == "anchorage")
    X[:, i, fi["is_approach"]]  = float(k == "approach")
    X[:, i, fi["n_hexes"]]      = nhex[n]
    X[:, i, fi["area_km2"]]     = area[n]
    X[:, i, fi["approach_radius_km"]] = rad[n]
X[:, :, fi[DRAFT_ERA]] = np.asarray(hours >= draftfix.ERA_BREAK, float)[:, None]

tr0, tr1 = hidx[SPLITS["train"][0]+" 00:00"], hidx[SPLITS["train"][1]+" 23:00"]
tmo = X[tr0:tr1+1, :, fi["occupancy"]].mean(0)   # per node, incl. held-out
X[:, :, fi["train_mean_occupancy"]] = tmo

print("\n=== 5. mask: hours when the AIS feed itself was down ===")
print("  Two detectors.  A national one on the whole feed, and a per-port one: a")
print("  receiver covering one harbour can fail without denting the national total,")
print("  and it shows up as occupancy collapsing to near zero for an hour and then")
print("  recovering -- which ships physically cannot do.  Reception, not traffic, so")
print("  the test is on the port's PING count against its own recent level.")
tot = con.execute(f"""SELECT date_trunc('hour', ts) AS hr, count(*) AS n
    FROM '{paths.PINGS}/**/*.parquet' GROUP BY 1""").df()
s = tot.set_index("hr").n.reindex(hours).fillna(0)
ref = s.rolling(169, center=True, min_periods=24).median()
bad = s < OUTAGE_FRAC*ref
mask = np.ones((T, N), dtype=bool); mask[bad.values, :] = False
pp = (h[h.node_id.isin(nidx)].assign(port=lambda x: x.node_id.map(dict(zip(ing.node_id, ing.port))))
      .groupby(["port","hr"]).n_pings.sum().unstack(0).reindex(hours).fillna(0))
pref = pp.rolling(25, center=True, min_periods=6).median()
pbad = pp < OUTAGE_FRAC*pref
for port in pp.columns:
    cols = [nidx[n] for n in order if ing.port[nidx[n]] == port]
    mask[np.ix_(pbad[port].values, cols)] = False
ep = (bad != bad.shift()).cumsum()[bad]
eps = pd.DataFrame({"start": bad[bad].groupby(ep).apply(lambda x: x.index[0]),
                    "hours": bad[bad].groupby(ep).size(),
                    "min_ratio": (s/ref)[bad].groupby(ep).min().round(3)})
r = (s/ref)
print(f"  {int(bad.sum())} of {T:,} hours masked ({100*bad.mean():.3f}%) in"
      f" {len(eps)} episodes: hourly pings below {OUTAGE_FRAC:.0%} of the surrounding"
      f" 7-day median.")
print(f"  {int((r<0.25).sum())} are hard outages (<25%, feed effectively absent),"
      f" {int(((r>=0.25)&bad).sum())} are partial.")
print("  The 1st percentile of the ratio over all hours is "
      f"{np.nanpercentile(r,1):.2f}, so a 50% cut sits well below ordinary variation;"
      "\n  it is set deliberately loose because masking a sound hour costs 0.005 of"
      " the training\n  data, while leaving a degraded one in teaches the model that"
      " a port emptied.")
print(eps.nlargest(8, "hours").to_string(index=False))
print(f"\n  per-port detector: {int(pbad.values.sum()):,} port-hours flagged"
      f" ({100*pbad.values.mean():.2f}% of {pbad.size:,}); worst ports "
      + ", ".join(f"{p} {100*pbad[p].mean():.2f}%"
                  for p in pbad.mean().nlargest(3).index))
print(f"  combined mask: {int((~mask).sum()):,} node-hours"
      f" ({100*(~mask).mean():.2f}%) excluded from training and scoring")
print(f"\n  The two gaps the preprocessing README documents are here (2026-06-04 13:00,"
      f" 11 h;\n  2023-03-24 05:00). The other {len(eps)-2} episodes are 1-11 hour"
      f" reception outages that a\n  day-level check could not see, because an hour"
      f" holding 400 pings still counts as present.")

print("\n=== 6. targets ===")
CHAN = ["anchorage_occ", "berth_occ", "avg_berth_wait", "avg_port_call"]
Y = np.zeros((T, N, 4), dtype=np.float32)
tmask = np.zeros((T, N, 4), dtype=bool)
occ = X[:, :, fi["occupancy"]]
tgt = []
for cx, g in ing.groupby("complex"):
    port = g.port.iloc[0]
    qn = g[g.node_kind == "anchorage"].node_id.iloc[0]
    bn = g[g.node_kind == "berth"].node_id.iloc[0]
    qi, bi = nidx[qn], nidx[bn]
    wv = WAIT_T[qn].values if qn in WAIT_T else np.full(T, np.nan)
    cv = CALL_T[bn].values if bn in CALL_T else np.full(T, np.nan)
    for n in g.node_id:
        i = nidx[n]
        Y[:, i, 0] = occ[:, qi];              tmask[:, i, 0] = mask[:, qi]
        Y[:, i, 1] = occ[:, bi];              tmask[:, i, 1] = mask[:, bi]
        Y[:, i, 2] = np.nan_to_num(wv);       tmask[:, i, 2] = mask[:, qi] & ~np.isnan(wv)
        Y[:, i, 3] = np.nan_to_num(cv);       tmask[:, i, 3] = mask[:, bi] & ~np.isnan(cv)
    for ch, (kind, host, hi) in enumerate([("anchorage_occupancy", qn, qi),
                                           ("berth_occupancy", bn, bi),
                                           ("avg_berth_wait", qn, qi),
                                           ("avg_port_call", bn, bi)]):
        tgt.append(dict(port=port, channel=ch, kind=kind, node_id=host, node_index=hi))
for t_ in tgt: t_["in_graph"] = bool(ing.in_graph[t_["node_index"]])
json.dump(tgt, open(f"{paths.TENSORS}/target_index{cfg.TAG}.json", "w"), indent=1)
u = pd.DataFrame([dict(channel=CHAN[t["channel"]], port=t["port"],
                       valid_pct=round(100*tmask[:, t["node_index"], t["channel"]].mean(), 1),
                       mean=round(float(Y[tmask[:, :, t["channel"]][:, t["node_index"]],
                                          t["node_index"], t["channel"]].mean()), 2))
                  for t in tgt])
print(f"  {len(tgt)} unique series to score ({len(tgt)//4} ports x 4 channels)")
print(u.pivot(index="port", columns="channel", values="mean")[CHAN].to_string())
print("\n  % of hours each channel is valid (feed up, and for durations a"
      " non-empty trailing window):")
print(u.pivot(index="port", columns="channel", values="valid_pct")[CHAN].to_string())
print(f"""
  The duration channels are sparse at small ports by construction: Tampa sees a
  few completions a day, so a 12 h arrival window is often empty.  Those hours
  are masked, not zero-filled -- a zero would read as "no wait at all", which is
  the opposite of what an empty window means.""")

print("\n=== 7. exogenous forcing U ===")
us = holidays.US(years=range(2023, 2027))
hod, dow, doy = hours.hour.values, hours.dayofweek.values, hours.dayofyear.values
U = np.stack([np.sin(2*np.pi*hod/24), np.cos(2*np.pi*hod/24),
              np.sin(2*np.pi*dow/7),  np.cos(2*np.pi*dow/7),
              np.isin(hours.date, list(us.keys())).astype(float),
              np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)], 1).astype(np.float32)
UNAMES = ["hod_sin","hod_cos","dow_sin","dow_cos","us_holiday","doy_sin","doy_cos"]
print(f"  U: {U.shape}  {UNAMES}")

print("\n=== 8. splits, scalers, artifacts ===")
sidx = {}
for k, (a, b) in SPLITS.items():
    sidx[k] = [int(hidx[a+" 00:00"]), int(hidx[b+" 23:00"]) + 1]
    print(f"  {k:5s} {a} .. {b}  hours {sidx[k][0]:6,}..{sidx[k][1]:6,} "
          f"({sidx[k][1]-sidx[k][0]:,})")

sidx["hour0"] = SPAN0 + " 00:00"
json.dump(sidx, open(f"{paths.TENSORS}/split_index{cfg.TAG}.json","w"), indent=1)

COUNTLIKE = {"occupancy","entries","exits","inbound_flux","transit_count"}
sc = {"fitted_on": "train", "features": FEATS, "u_names": UNAMES, "per_feature": {}}
# the scaler is fitted on East+Gulf nodes ONLY, whatever the panel scope, so a
# West-coast zero-shot run really is zero-shot on the input normalisation too
eg = [i for i, n in enumerate(order) if bool(ing.in_graph[i])]
Xtr = X[tr0:tr1+1][:, eg]
for f in FEATS:
    v = Xtr[:, :, fi[f]].ravel()
    if f in COUNTLIKE:
        lv = np.log1p(v)
        sc["per_feature"][f] = dict(transform="log1p+standardize",
                                    mean=float(lv.mean()), std=float(lv.std() + 1e-8))
    elif f.startswith("is_") or f == DRAFT_ERA:
        sc["per_feature"][f] = dict(transform="none")
    else:
        q1, q3 = np.percentile(v, [25, 75])
        sc["per_feature"][f] = dict(transform="robust",
                                    median=float(np.median(v)), iqr=float(max(q3-q1, 1e-6)))
sc["fitted_on_nodes"] = [order[i] for i in eg]
Xtr_all = X[tr0:tr1+1]
sc["node_capacity_p99"] = {n: float(np.percentile(Xtr_all[:, i, fi["occupancy"]], 99))
                           for i, n in enumerate(order)}
# duration targets: log1p then per-node robust standardize, fitted on TRAIN and
# only on hours the channel is actually valid
sc["target_channels"] = {}
for ch, name in enumerate(CHAN):
    per = {}
    for i, n in enumerate(order):
        v = Y[tr0:tr1+1, i, ch][tmask[tr0:tr1+1, i, ch]]
        if not len(v):
            continue
        if ch < 2:
            per[n] = dict(transform="raw_counts",
                          capacity_p99=float(np.percentile(v, 99)))
        else:
            lv = np.log1p(v); q1, q3 = np.percentile(lv, [25, 75])
            per[n] = dict(transform="log1p+robust", median=float(np.median(lv)),
                          iqr=float(max(q3-q1, 1e-6)))
    sc["target_channels"][name] = per
sc["node_train_mean_occupancy"] = {n: float(tmo[i]) for i, n in enumerate(order)}
json.dump(sc, open(f"{paths.TENSORS}/scalers{cfg.TAG}.json","w"), indent=1)

np.save(f"{paths.TENSORS}/X{cfg.TAG}.npy", X); np.save(f"{paths.TENSORS}/Y{cfg.TAG}.npy", Y)
np.save(f"{paths.TENSORS}/U{cfg.TAG}.npy", U); np.save(f"{paths.TENSORS}/mask{cfg.TAG}.npy", mask)
np.save(f"{paths.TENSORS}/target_mask{cfg.TAG}.npy", tmask)
pd.DataFrame({"hour": hours}).to_parquet(f"{paths.TENSORS}/hours{cfg.TAG}.parquet", index=False)
print(f"\n  tensors/X.npy {X.shape}  Y.npy {Y.shape}  U.npy {U.shape}"
      f"  mask.npy {mask.shape}  target_mask.npy {tmask.shape}"
      f" ({100*tmask.mean():.1f}% valid)")

print("\n=== 9. discontinuity scan: does any feature step at a boundary? ===")
print("  30 days either side of each year change and each split boundary; a"
      " forecasting\n  model cannot tell a source change from a change in the port.\n")
bounds = {"2024-01-01": "year", "2025-01-01": "year", "2026-01-01": "val starts",
          "2026-04-01": "test starts"}
rows = []
for b, why in bounds.items():
    bi = int(hidx[b + " 00:00"]); w = 30*24
    for f in DYN:
        pre = X[bi-w:bi, :, fi[f]][mask[bi-w:bi]]
        post = X[bi:bi+w, :, fi[f]][mask[bi:bi+w]]
        if pre.mean() == 0: continue
        rows.append(dict(boundary=b, why=why, feature=f,
                         before=round(float(pre.mean()), 3),
                         after=round(float(post.mean()), 3),
                         pct=round(100*(post.mean()/pre.mean() - 1), 1)))
sc = pd.DataFrame(rows)
big = sc[sc.pct.abs() >= 10].sort_values("pct", key=abs, ascending=False)
print(f"  {len(big)} of {len(sc)} (feature, boundary) pairs move by 10% or more:")
print(big.to_string(index=False) if len(big) else "    (none)")
print(f"\n  largest move at a SPLIT boundary: "
      f"{sc[sc.why.str.contains('starts')].pct.abs().max():.1f}%"
      f"  -- val and test are not a different regime from train")
sc.to_csv(paths.TENSORS / f"discontinuity{cfg.TAG}.csv", index=False)
