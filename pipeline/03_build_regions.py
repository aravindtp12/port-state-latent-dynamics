"""Freeze the graph once, on training data only.

The node geometry is fitted on 2023-01-01 .. 2025-12-24 -- the training period
less a 7-day buffer -- and then held fixed for val (Jan-Mar 2026) and test
(Apr-Jun 2026).  Two reasons:

  * the anchorage rule looks 7 days FORWARD (a patch of water belongs to port P
    if the ships that stop there go on to berth at P), so fitting it on data
    that touches val/test would leak the future into training;
  * the same forward look silently deletes hexagons first used in the last week
    of whatever window it is fitted on.  Ending the fit 7 days early removes
    that right-edge artefact from the frozen geometry, and inference then uses
    a fixed hexagon set with no per-window lookahead at all.

Region rules: East+Gulf is the graph; the West coast is a
disconnected coverage region (2 of 1,169 voyages cross the gap) kept aside as a
zero-shot transfer set; Sault Ste Marie is a lock queue, not a port call, and is
dropped; Houston and Galveston are merged -- Galveston is Houston's queue.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from psld import paths, config as cfg
paths.ensure(paths.DUCKDB_TMP, paths.GRAPH, paths.WORK, paths.TENSORS, paths.RESULTS)
import duckdb, numpy as np, pandas as pd, h3, json
from psld import geoutil as G
pd.set_option("display.width", 250, "display.max_columns", 60)

cfg.banner()
FIT0, FIT1 = cfg.SPAN0, cfg.FIT_END          # training period less a 7-day buffer
STAT_PING_S = 180.0                            # measured stationary ping interval
BERTH_KM, ASSIGN_KM, MAX_SERVE_KM = 0.5, 100.0, 100.0
GAP_MIN, LOOKAHEAD_D, MIN_SHARE = 120, 7, 0.5
KEEP_SHARE = 0.95   # the seed threshold is whatever retains this share of the
                    # stationary vessel-hours inside the port discs -- see below
N_FIT_DAYS = (pd.Timestamp(FIT1) - pd.Timestamp(FIT0)).days + 1

WEST  = {4, 7, 9}                              # Vancouver, Oakland, Long Beach
LAKES = {39}                                   # Sault Ste Marie
MERGE = {20: 19}                               # Galveston -> Houston
GULF  = {19, 20, 17, 16, 21, 15}
EAST  = {14, 30, 34, 36, 35, 38}

print(f"=== 0. fit window {FIT0} .. {FIT1} ({N_FIT_DAYS} days) ===")
print(f"  seed threshold: the largest T retaining {KEEP_SHARE:.0%} of the stationary"
      f" vessel-hours inside the port discs")

# 16 reference points (one per port; Galveston is merged into Houston below)
ports = pd.read_csv(paths.PORTS_CSV)[["label","complex","ref_lon","ref_lat"]]
ports["merged"] = ports["complex"].replace(MERGE)
ports["region"] = np.where(ports["complex"].isin(WEST), "W",
                  np.where(ports["complex"].isin(LAKES), "L",
                  np.where(ports["complex"].isin(GULF), "G", "E")))
print(f"  {len(ports)} reference points -> {ports.merged.nunique()} complexes after merging"
      f" Galveston into Houston")

# ---------------------------------------------------------------- port discs
con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';"
            f"PRAGMA temp_directory='{paths.DUCKDB_TMP}'; PRAGMA disable_progress_bar")
con.execute("INSTALL h3 FROM community; LOAD h3;")
con.execute(f"CREATE VIEW p AS SELECT * FROM '{paths.PINGS}/**/*.parquet' "
            f"WHERE day BETWEEN DATE '{FIT0}' AND DATE '{FIT1}'")

step7 = h3.average_hexagon_edge_length(7, unit="km") * np.sqrt(3)   # centre-to-centre
kring = int(np.ceil(ASSIGN_KM / step7)) + 1
disc = set()
for _, r in ports.iterrows():
    c0 = h3.latlng_to_cell(r.ref_lat, r.ref_lon, 7)
    for c in h3.grid_disk(c0, kring):
        la, lo = h3.cell_to_latlng(c)
        if np.hypot((lo-r.ref_lon)*111.32*np.cos(np.radians(la)),
                    (la-r.ref_lat)*110.574) <= ASSIGN_KM:
            disc.add(c)
disc = pd.DataFrame({"cell7": np.array([h3.str_to_int(c) for c in sorted(disc)],
                                       dtype="uint64")})
print(f"  res-7 prefilter: {len(disc):,} cells cover every port disc of {ASSIGN_KM:.0f} km")
con.register("disc", disc)

print("\n=== 1. stationary pings -> H3 res-10 hexagons (fit window) ===")
hx = con.execute("""
SELECT h3_10, count(*) AS n, count(DISTINCT mmsi) AS v,
       count(*) FILTER (status = 5) AS moored,
       count(*) FILTER (status = 1) AS anchored,
       min(ts) AS first_ts, max(ts) AS last_ts
FROM p SEMI JOIN disc d ON h3_cell_to_parent(h3_10, 7) = d.cell7
WHERE stationary GROUP BY h3_10""").df()
hx["h10"] = [h3.int_to_str(int(c)) for c in hx.h3_10]
ll = np.array([h3.cell_to_latlng(c) for c in hx.h10])
hx["lat"], hx["lon"] = ll[:, 0], ll[:, 1]
hx["vessel_hours"] = hx.n * STAT_PING_S / 3600.0
print(f"  {len(hx):,} hexagons, {hx.n.sum():,} stationary pings, "
      f"{hx.vessel_hours.sum():,.0f} stationary vessel-hours")

d = np.hypot((hx.lon.values[:, None] - ports.ref_lon.values) * 111.32
             * np.cos(np.radians(hx.lat.values[:, None])),
             (hx.lat.values[:, None] - ports.ref_lat.values) * 110.574)
j = d.argmin(1)
hx["complex"] = ports.merged.values[j]
hx["port_km"] = d.min(1)
hx = hx[hx.port_km <= ASSIGN_KM].reset_index(drop=True)
lbl = (ports.drop_duplicates("merged").set_index("merged").label
       .str.split(" (", regex=False).str[0].replace({"Houston": "Houston-Galveston"}))
reg = ports.drop_duplicates("merged").set_index("merged").region
hx["label"] = hx["complex"].map(lbl)
print(f"  within {ASSIGN_KM:.0f} km of a port: {len(hx):,} hexagons, {hx.n.sum():,} pings")

# ---------------------------------------------------------------- seed sweep
print(f"\n=== 2. seed threshold ===")
print("  An absolute vessel-hour threshold does not transfer between window lengths: berths are")
print("  the same few hexagons used again and again, while an anchorage is a scatter of one-off")
print("  stopping points that spreads wider the longer the window, so a threshold tuned on a")
print("  short window deletes anchorages on a long one.  So the RETENTION is fixed, not the number:")
print(f"  T is the largest threshold that keeps {KEEP_SHARE:.0%} of the stationary")
print("  vessel-hours inside the port discs, which means the same thing at any window length.\n")
vh = np.sort(hx.vessel_hours.values)[::-1]
cum = np.cumsum(vh) / vh.sum()
T_SCALED = float(vh[np.searchsorted(cum, KEEP_SHARE)])
print(f"  -> T = {T_SCALED:.1f} vessel-hours "
      f"({int((hx.vessel_hours >= T_SCALED).sum()):,} of {len(hx):,} hexagons)")
T_MIN = T_SCALED * 0.5
base0 = hx[hx.vessel_hours >= T_MIN].copy()
kind, signed = G.classify(base0.lon.values, base0.lat.values)
base0["shore_km"] = np.abs(signed); base0["land"] = (kind == "inland")
base0["moored_share"] = base0.moored / base0.n
base0["anchor_share"] = base0.anchored / base0.n

def classify_nodes(b):
    geo_berth = b.land | (b.shore_km < BERTH_KM)
    to_anchor = geo_berth & (b.anchor_share > 0.6)
    to_berth  = ~geo_berth & (b.shore_km < 1.5) & (b.moored_share > 0.6)
    return np.where((geo_berth & ~to_anchor) | to_berth, "berth", "anchorage")

rows = []
for mult in (0.5, 1.0, 2.0, 5.0, 10.0, 176.0/T_SCALED):
    T = T_SCALED * mult
    b = base0[base0.vessel_hours >= T].copy()
    b["node"] = classify_nodes(b)
    agb = [100*g.moored.sum()/g.n.sum() for (_c, nd), g in b.groupby(["complex","node"])
           if nd == "berth" and len(g) >= 3]
    aga = [100*g.anchored.sum()/g.n.sum() for (_c, nd), g in b.groupby(["complex","node"])
           if nd == "anchorage" and len(g) >= 3]
    rows.append(dict(multiplier=round(mult, 2), T_vessel_hours=round(T, 1), seed_hexes=len(b),
                     pct_stationary_pings_kept=round(100*b.n.sum()/hx.n.sum(), 1),
                     berth_seeds=int((b.node=="berth").sum()),
                     anchorage_seeds=int((b.node=="anchorage").sum()),
                     median_berth_agree=round(np.median(agb), 1),
                     median_anch_agree=round(np.median(aga), 1)))
sweep = pd.DataFrame(rows); sweep.to_csv(paths.GRAPH / "seed_sweep.csv", index=False)
print(sweep.to_string(index=False))

T = T_SCALED
print(f"\n=== 3. berth nodes at T = {T:.0f} vessel-hours ===")
base = base0[base0.vessel_hours >= T].copy()
base["node"] = classify_nodes(base)
geo_berth = base.land | (base.shore_km < BERTH_KM)
print(f"  geometric berth {int(geo_berth.sum()):,} (inside land polygon {int(base.land.sum()):,},"
      f" within {BERTH_KM} km of shore {int((~base.land & (base.shore_km<BERTH_KM)).sum()):,})")
print(f"  status overrides: -> anchorage {int((geo_berth & (base.anchor_share>0.6)).sum()):,},"
      f" -> berth {int((~geo_berth & (base.shore_km<1.5) & (base.moored_share>0.6)).sum()):,}")
print(f"  seeds: berth {int((base.node=='berth').sum()):,}, "
      f"anchorage {int((base.node=='anchorage').sum()):,}")

def close_h3(cells):
    """Morphological closing on the H3 lattice: dilate one ring, then erode."""
    s = set(cells); dil = set()
    for c in s:
        dil |= set(h3.grid_disk(c, 1))
    return {c for c in dil if set(h3.grid_disk(c, 1)) <= dil}

berth = []
for cx, g in base[base.node == "berth"].groupby("complex"):
    if len(g) < 3: continue
    cells = sorted(close_h3(g.h10.tolist()))
    berth.append(pd.DataFrame({"complex": cx, "label": lbl[cx], "node": "berth", "cell": cells}))
berth = pd.concat(berth, ignore_index=True)
print(f"  after closing: {len(berth):,} berth hexagons across {berth.complex.nunique()} complexes")

# ------------------------------------------------- anchorage by destination
print(f"\n=== 4. anchorage nodes, attached by observed destination ===")
anch0 = base[base.node == "anchorage"]
con.register("berth_hex", berth[["cell","complex"]])
con.register("anch_hex", anch0[["h10"]].rename(columns={"h10":"cell"}).drop_duplicates())
con.execute("CREATE OR REPLACE VIEW sp AS "
            "SELECT mmsi, ts, h3_h3_to_string(h3_10) AS cell FROM p WHERE stationary")
con.execute(f"""
CREATE OR REPLACE TABLE berth_spells AS
WITH j AS (SELECT s.mmsi, s.ts, b.complex FROM sp s JOIN berth_hex b USING (cell)),
 f AS (SELECT *, CASE WHEN date_diff('minute', lag(ts) OVER w, ts) > {GAP_MIN}
         OR lag(ts) OVER w IS NULL THEN 1 ELSE 0 END AS ns
       FROM j WINDOW w AS (PARTITION BY mmsi, complex ORDER BY ts)),
 g AS (SELECT *, sum(ns) OVER (PARTITION BY mmsi, complex ORDER BY ts) AS sid FROM f)
SELECT mmsi, complex, min(ts) AS t_in, max(ts) AS t_out, count(*) AS n
FROM g GROUP BY mmsi, complex, sid""")
print("  berth spells in the fit window: " + con.execute(
    "SELECT count(*)::VARCHAR || ' spells, ' || count(DISTINCT mmsi)::VARCHAR || ' vessels'"
    " FROM berth_spells").fetchone()[0])

# next berth call per anchorage-candidate stay, in one pass:  interleave stays and
# berth-spell starts on a per-vessel timeline and carry the next berth backwards.
attr = con.execute(f"""
WITH stays AS (
  WITH j AS (SELECT s.mmsi, s.ts, s.cell FROM sp s SEMI JOIN anch_hex a USING (cell)),
   f AS (SELECT *, CASE WHEN date_diff('minute', lag(ts) OVER w, ts) > {GAP_MIN}
           OR lag(ts) OVER w IS NULL THEN 1 ELSE 0 END AS ns
         FROM j WINDOW w AS (PARTITION BY mmsi, cell ORDER BY ts)),
   g AS (SELECT *, sum(ns) OVER (PARTITION BY mmsi, cell ORDER BY ts) AS sid FROM f)
  SELECT mmsi, cell, min(ts) AS t_in, max(ts) AS t_out, count(*) AS n
  FROM g GROUP BY mmsi, cell, sid),
 ev AS (
  SELECT mmsi, t_out AS t, cell, n, NULL::INTEGER AS bcx FROM stays
  UNION ALL
  SELECT mmsi, t_in AS t, NULL, NULL, complex FROM berth_spells),
 w AS (
  SELECT *, first_value(bcx IGNORE NULLS) OVER nx AS next_complex,
            first_value(CASE WHEN bcx IS NOT NULL THEN t END IGNORE NULLS) OVER nx AS next_t
  FROM ev WINDOW nx AS (PARTITION BY mmsi ORDER BY t
                        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING))
SELECT cell, CASE WHEN next_t <= t + INTERVAL {LOOKAHEAD_D} DAY THEN next_complex END
         AS next_complex,
       sum(n) * {STAT_PING_S}/3600.0 AS vessel_hours
FROM w WHERE cell IS NOT NULL
  AND t <= TIMESTAMP '{FIT1} 23:59:59' - INTERVAL {LOOKAHEAD_D} DAY
GROUP BY 1, 2""").df()

tot = attr.groupby("cell").vessel_hours.sum().rename("total_h")
named = attr[attr.next_complex.notna()].copy()
named["next_complex"] = named.next_complex.astype(int)
refs = ports.groupby("merged")[["ref_lon","ref_lat"]].mean()
cll = np.array([h3.cell_to_latlng(c) for c in named.cell])
rl = refs.loc[named.next_complex.values][["ref_lat","ref_lon"]].values
named["serve_km"] = np.hypot((cll[:,1]-rl[:,1])*111.32*np.cos(np.radians(cll[:,0])),
                             (cll[:,0]-rl[:,0])*110.574)
far = named.serve_km > MAX_SERVE_KM
print(f"  dropped {int(far.sum()):,} (hexagon, destination) pairs beyond {MAX_SERVE_KM:.0f} km"
      f" ({named.vessel_hours[far].sum():,.0f} vessel-hours)")
named = named[~far]
best = named.loc[named.groupby("cell").vessel_hours.idxmax()].set_index("cell")
res = best.join(tot, how="left"); res["share"] = res.vessel_hours / res.total_h
keep = res[res.share >= MIN_SHARE].reset_index()
dropped_h = tot.loc[tot.index.difference(keep.cell)].sum()
print(f"  candidates {len(tot):,} hexagons, {tot.sum():,.0f} vessel-hours")
print(f"  attached ({MIN_SHARE:.0%} of stationary vessel-hours followed by a berth call at that"
      f" port within {LOOKAHEAD_D} d): {len(keep):,}")
print(f"  dropped - their ships do not berth at any shortlisted port: "
      f"{len(tot)-len(keep):,} hexagons, {dropped_h:,.0f} vessel-hours "
      f"({100*dropped_h/tot.sum():.1f}%)")

old = anch0.drop_duplicates("h10").set_index("h10")["complex"]
keep = keep.merge(old.rename("old_complex"), left_on="cell", right_index=True, how="left")
mv = keep[keep.next_complex != keep.old_complex]
if len(mv):
    t = (mv.assign(frm=mv.old_complex.map(lbl), to=mv.next_complex.map(lbl))
           .groupby(["frm","to"]).agg(hexagons=("cell","size"),
                                      vessel_hours=("vessel_hours","sum")).reset_index())
    print("\n  reassignments (nearest port -> port whose berths its ships use):")
    print(t.sort_values("vessel_hours", ascending=False).head(10).round(0).to_string(index=False))

recs = [berth]
for cx, g in keep.groupby("next_complex"):
    cells = sorted(close_h3(g.cell.tolist()))
    recs.append(pd.DataFrame({"complex": cx, "label": lbl[cx], "node": "anchorage",
                              "cell": cells}))
nodes_hex = pd.concat(recs, ignore_index=True)
ll = np.array([h3.cell_to_latlng(c) for c in nodes_hex.cell])
nodes_hex["lat"], nodes_hex["lon"] = ll[:,0], ll[:,1]
print(f"\n  final: {int((nodes_hex.node=='berth').sum()):,} berth hexagons, "
      f"{int((nodes_hex.node=='anchorage').sum()):,} anchorage hexagons")

# ---------------------------------------------------------------- approach R
print("\n=== 5. approach radii ===")
area1 = h3.average_hexagon_area(10, unit="km^2")
recs = []
for cx, g in nodes_hex.groupby("complex"):
    b, a = g[g.node == "berth"], g[g.node == "anchorage"]
    rlat, rlon = b.lat.mean(), b.lon.mean()
    cos = np.cos(np.radians(rlat))
    if len(a):
        da = np.hypot((a.lon-rlon)*111.32*cos, (a.lat-rlat)*110.574)
        p95 = float(np.percentile(da, 95))
    else:
        p95 = 0.0
    R = float(np.clip(10*np.ceil((p95+5)/10), 20.0, 100.0))
    for kind in ("berth", "anchorage", "approach"):
        sub = g[g.node == kind]
        recs.append(dict(port=lbl[cx], complex=int(cx), node_kind=kind,
                         region=reg[cx],
                         h3_cells=sorted(sub.cell.tolist()) if kind != "approach" else [],
                         n_hexes=len(sub),
                         area_km2=round(len(sub)*area1, 3) if kind != "approach"
                                  else round(np.pi*R*R, 1),
                         centroid_lat=round(float(sub.lat.mean()) if len(sub) else rlat, 5),
                         centroid_lon=round(float(sub.lon.mean()) if len(sub) else rlon, 5),
                         ref_lat=round(float(rlat), 5), ref_lon=round(float(rlon), 5),
                         approach_radius_km=R if kind == "approach" else np.nan,
                         anchorage_p95_km=round(p95, 1)))
nodes = pd.DataFrame(recs)
nodes["in_graph"] = nodes.region.isin(["E","G"])
nodes = nodes.sort_values(["in_graph","region","port","node_kind"],
                          ascending=[False,True,True,True]).reset_index(drop=True)
SUF = {"berth": "B", "anchorage": "Q", "approach": "A"}   # B_p, Q_p, A_p
nodes.insert(0, "node_id", [f"{r.port.replace(' ','').replace('-','')}_{SUF[r.node_kind]}"
                            for r in nodes.itertuples()])
assert nodes.node_id.is_unique, "node_id collision"
chk = nodes[nodes.node_kind=="approach"][["port","region","in_graph","approach_radius_km",
                                          "anchorage_p95_km"]]
print(chk.to_string(index=False))

nodes.to_parquet(f"{paths.GRAPH}/nodes.parquet")
print(f"\n  -> graph/nodes.parquet: {len(nodes)} nodes "
      f"({int(nodes.in_graph.sum())} in the East+Gulf graph, "
      f"{int((~nodes.in_graph).sum())} held out)")
print(nodes.groupby(["in_graph","region"]).size().rename("nodes").to_string())

# ---------------------------------------------------------------- coverage
print("\n=== 6. coverage check on the frozen geometry (fit window) ===")
cellmap = pd.concat([pd.DataFrame({"cell": r.h3_cells, "node_id": r.node_id,
                                   "complex": r["complex"], "node_kind": r.node_kind})
                     for _, r in nodes.iterrows() if len(r.h3_cells)], ignore_index=True)
cellmap.to_parquet(f"{paths.GRAPH}/node_cells.parquet")
con.register("nh", cellmap)
# each hexagon belongs to its nearest complex, so overlapping port discs (Houston
# and Beaumont, Newark and Gloucester City) cannot double-count the same ping
disc_hex = hx[["h10","complex","port_km"]].rename(columns={"h10":"cell"})
cm = cellmap[["cell","node_kind"]]
cv = (disc_hex.merge(cm, on="cell", how="left")
      .merge(hx[["h10","n"]].rename(columns={"h10":"cell","n":"pings"}), on="cell"))
cv["node_kind"] = cv.node_kind.fillna("outside")
tab = (cv.pivot_table(index="complex", columns="node_kind", values="pings",
                      aggfunc="sum", fill_value=0))
for c in ("berth","anchorage","outside"):
    if c not in tab: tab[c] = 0
tab["captured_pct"] = (100*(tab.berth+tab.anchorage)/tab.sum(axis=1)).round(1)
tab["port"] = [lbl[i] for i in tab.index]
tab["in_graph"] = [reg[i] in ("E","G") for i in tab.index]
tab = tab.sort_values(["in_graph","captured_pct"], ascending=[False,False])
print(tab[["port","in_graph","berth","anchorage","outside","captured_pct"]].to_string(index=False))
ing = tab[tab.in_graph]
print(f"\n  median capture, East+Gulf: {ing.captured_pct.median():.1f}%"
      "")
print("  pings outside every node are stops at the ~115 other US port clusters the"
      "\n  16-port shortlist deliberately excludes, plus mid-transit stops offshore")
tab.to_csv(paths.GRAPH / "coverage.csv", index=False)
