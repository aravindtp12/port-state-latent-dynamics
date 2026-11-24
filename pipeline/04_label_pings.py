"""Apply the frozen graph to every ping, 2023-01-01 .. 2026-06-30.

Runs month by month so nothing large is ever resident, and writes three
intermediate tables that everything downstream is built from:

  work/hourly/    per (node, hour): occupancy, fleet composition, draft
  work/buckets/   stationary node pings reduced to 30-minute buckets, keeping
                  exact min/max timestamps -- spells split on gaps > 2 h, so a
                  30-minute bucket can never hide a split and t_in / t_out stay
                  exact.  ~25x smaller than the raw pings.
  work/crossings/ inbound crossings of each approach ring

Node membership: berth and anchorage are exact H3 res-10 lookups; approach is
everything else inside radius R of the port, by true distance, with a ping that
falls inside two ports' rings going to the nearer one.
"""
import os, time, shutil
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from psld import paths, config as cfg
paths.ensure(paths.DUCKDB_TMP, paths.GRAPH, paths.WORK, paths.TENSORS, paths.RESULTS)
import duckdb, numpy as np, pandas as pd, h3
from psld.sqlutil import haversine_km

SPAN0, SPAN1 = "2023-01-01", "2026-06-30"
BUCKET_S, OUTER_KM, MAX_CROSS_GAP_H = 1800, 40.0, 6.0
WORK = str(paths.WORK)
for sub in ("hourly", "buckets", "crossings"):
    d = f"{WORK}/{sub}"
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)

nodes = pd.read_parquet(f"{paths.GRAPH}/nodes.parquet")
cells = pd.read_parquet(f"{paths.GRAPH}/node_cells.parquet")
cells["cell_int"] = [h3.str_to_int(c) for c in cells.cell]
ring = nodes[nodes.node_kind == "approach"][["complex","node_id","ref_lon","ref_lat",
                                             "approach_radius_km"]].copy()
ring = ring.rename(columns={"approach_radius_km": "R"})
print(f"{len(nodes)} nodes, {len(cells):,} berth/anchorage hexagons, "
      f"{len(ring)} approach rings (R {ring.R.min():.0f}-{ring.R.max():.0f} km)")

# res-7 discs of radius R + OUTER_KM: the outer band is needed to see a vessel
# on both sides of the ring and so detect the crossing itself
step7 = h3.average_hexagon_edge_length(7, unit="km") * np.sqrt(3)
rows = []
for r in ring.itertuples():
    lim = r.R + OUTER_KM
    c0 = h3.latlng_to_cell(r.ref_lat, r.ref_lon, 7)
    for c in h3.grid_disk(c0, int(np.ceil(lim/step7)) + 1):
        la, lo = h3.cell_to_latlng(c)
        if np.hypot((lo-r.ref_lon)*111.32*np.cos(np.radians(la)),
                    (la-r.ref_lat)*110.574) <= lim:
            rows.append((h3.str_to_int(c), r.complex))
disc = pd.DataFrame(rows, columns=["cell7","complex"]).astype({"cell7":"uint64"})
disc = disc.merge(ring[["complex","ref_lon","ref_lat","R"]], on="complex")
print(f"res-7 prefilter: {len(disc):,} (cell, port) pairs over {disc.cell7.nunique():,} cells")

ves = pd.read_parquet(paths.VESSELS)[["mmsi","vessel_type","length"]]
ves["is_cargo"] = ves.vessel_type.between(70, 79)

con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='11GB';"
            f"PRAGMA temp_directory='{paths.DUCKDB_TMP}'; PRAGMA disable_progress_bar")
con.execute("INSTALL h3 FROM community; LOAD h3;")
con.register("nc", cells[["cell_int","node_id","complex"]])
con.register("disc", disc)
con.register("ves", ves)
con.register("ringtab", ring[["complex","node_id"]].rename(columns={"node_id":"ring_node"}))

months = pd.period_range(SPAN0, SPAN1, freq="M")
t0 = time.time(); tot_h = tot_b = tot_c = 0
for i, m in enumerate(months):
    d0, d1 = m.start_time.date(), m.end_time.date()
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE lab AS
    WITH base AS (
      SELECT mmsi, ts, lon, lat, draft, stationary, h3_10,
             h3_cell_to_parent(h3_10, 7) AS c7
      FROM '{paths.PINGS}/**/*.parquet'
      WHERE day BETWEEN DATE '{d0}' AND DATE '{d1}'),
    -- exact hexagon lookup first: a berth or anchorage ping is never ambiguous
    b2 AS (SELECT b.*, nc.node_id AS hex_node FROM base b
           LEFT JOIN nc ON b.h3_10 = nc.cell_int),
    -- then the rings, with true distance; the outer band feeds crossings only
    rg AS (SELECT b2.mmsi, b2.ts, b2.draft, b2.stationary, b2.hex_node,
                  d.complex, d.R,
                  {haversine_km("b2.lon","b2.lat","d.ref_lon","d.ref_lat")} AS dist
           FROM b2 JOIN disc d ON b2.c7 = d.cell7),
    rk AS (SELECT *, row_number() OVER (PARTITION BY mmsi, ts ORDER BY dist) AS rn
           FROM rg WHERE dist <= R)
    SELECT rk.mmsi, rk.ts, rk.draft, rk.stationary,
           coalesce(rk.hex_node, rt.ring_node) AS node_id
    FROM rk JOIN ringtab rt USING (complex)
    WHERE rk.rn = 1 OR rk.hex_node IS NOT NULL""")
    # a hexagon ping keeps its own node even when another port's ring is nearer,
    # so de-duplicate to one row per (mmsi, ts) preferring the hexagon label
    con.execute("""
    CREATE OR REPLACE TEMP TABLE lab2 AS
    SELECT mmsi, ts, draft, stationary, node_id FROM (
      SELECT *, row_number() OVER (PARTITION BY mmsi, ts ORDER BY node_id) AS r2
      FROM lab) WHERE r2 = 1""")

    con.execute(f"""COPY (
      WITH hv AS (
        SELECT node_id, date_trunc('hour', ts) AS hr, mmsi,
               max(stationary::INT) AS stat, avg(draft) AS draft, count(*) AS np
        FROM lab2 GROUP BY 1,2,3)
      SELECT h.node_id, h.hr, sum(h.np) AS n_pings,
             count(*) FILTER (h.stat = 1) AS occupancy,
             count(*) FILTER (h.stat = 0) AS transit_count,
             avg(h.draft)   FILTER (h.stat = 1) AS mean_draft,
             avg(v.length)  FILTER (h.stat = 1) AS mean_length,
             max(v.length)  FILTER (h.stat = 1) AS max_length,
             count(*) FILTER (h.stat = 1 AND v.is_cargo) AS n_cargo,
             count(*) FILTER (h.stat = 1 AND NOT v.is_cargo) AS n_tanker
      FROM hv h LEFT JOIN ves v USING (mmsi) GROUP BY 1,2)
      TO '{WORK}/hourly/{m}.parquet' (COMPRESSION zstd)""")

    con.execute(f"""COPY (
      SELECT mmsi, node_id, epoch(ts)::BIGINT // {BUCKET_S} AS bucket,
             min(ts) AS t_min, max(ts) AS t_max, count(*) AS n, median(draft) AS draft
      FROM lab2 WHERE stationary GROUP BY 1,2,3)
      TO '{WORK}/buckets/{m}.parquet' (COMPRESSION zstd)""")

    # ring crossings both ways: consecutive pings on opposite sides of R, close
    # enough in time to be one continuous movement rather than a reception gap
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE rgall AS
    WITH base AS (
      SELECT mmsi, ts, lon, lat, h3_cell_to_parent(h3_10, 7) AS c7
      FROM '{paths.PINGS}/**/*.parquet'
      WHERE day BETWEEN DATE '{d0}' - INTERVAL 1 DAY AND DATE '{d1}')
    SELECT b.mmsi, b.ts, d.complex, d.R,
           {haversine_km("b.lon","b.lat","d.ref_lon","d.ref_lat")} AS dist
    FROM base b JOIN disc d ON b.c7 = d.cell7""")
    con.execute(f"""COPY (
      WITH s AS (SELECT *, lag(dist) OVER w AS dprev, lag(ts) OVER w AS tprev
                 FROM rgall WINDOW w AS (PARTITION BY mmsi, complex ORDER BY ts))
      SELECT rt.ring_node AS node_id, s.mmsi, s.ts AS t_cross,
             CASE WHEN s.dist <= s.R THEN 'in' ELSE 'out' END AS direction
      FROM s JOIN ringtab rt USING (complex)
      WHERE ((s.dprev > s.R AND s.dist <= s.R) OR (s.dprev <= s.R AND s.dist > s.R))
        AND date_diff('minute', s.tprev, s.ts) <= {int(MAX_CROSS_GAP_H*60)}
        AND s.ts >= TIMESTAMP '{d0} 00:00:00')
      TO '{WORK}/crossings/{m}.parquet' (COMPRESSION zstd)""")

    nh, nb, ncr = (con.execute(
        f"SELECT (SELECT count(*) FROM '{WORK}/hourly/{m}.parquet'),"
        f" (SELECT count(*) FROM '{WORK}/buckets/{m}.parquet'),"
        f" (SELECT count(*) FROM '{WORK}/crossings/{m}.parquet')").fetchone())
    tot_h += nh; tot_b += nb; tot_c += ncr
    print(f"  {m}  hourly {nh:6,}  buckets {nb:9,}  crossings {ncr:5,}"
          f"   [{time.time()-t0:5.0f}s]", flush=True)

print(f"\ntotal: {tot_h:,} node-hours, {tot_b:,} buckets, {tot_c:,} ring crossings"
      f"  in {time.time()-t0:.0f}s")
for sub in ("hourly","buckets","crossings"):
    sz = sum(os.path.getsize(f"{WORK}/{sub}/{f}") for f in os.listdir(f"{WORK}/{sub}"))
    print(f"  work/{sub}: {sz/1e6:.0f} MB")
