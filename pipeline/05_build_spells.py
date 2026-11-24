"""Spells and the per-vessel duration event table.

A spell is one continuous stay by one vessel in one node, split whenever the
vessel goes unseen in that node for more than 2 hours.  Built from the 30-minute
buckets of step 4, which preserves both the gap structure and the exact endpoints.

Two duration targets come out of it:
  port_call  = length of a berth spell.
  berth_wait = length of an anchorage spell.  It is only an observed waiting
               time if the vessel then berths at the same port; 43% of
               anchorage spells end with the vessel leaving instead.
               That is a COMPETING RISK, not missing data, so the
               row is kept with event_observed = False and exit_reason = 'left'.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from psld import paths, config as cfg
paths.ensure(paths.DUCKDB_TMP, paths.GRAPH, paths.WORK, paths.TENSORS, paths.RESULTS)
import duckdb, numpy as np, pandas as pd
from psld import draftfix
pd.set_option("display.width", 250, "display.max_columns", 60)
SPAN0, SPAN1 = "2023-01-01 00:00:00", "2026-06-30 23:59:59"
GAP_MIN, EDGE_H, BERTH_WITHIN_D, LONG_STAY_D = 120, 12, 3, 30

nodes = pd.read_parquet(f"{paths.GRAPH}/nodes.parquet")[
    ["node_id","port","complex","node_kind","region","in_graph"]]
con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='11GB';"
            f"PRAGMA temp_directory='{paths.DUCKDB_TMP}'; PRAGMA disable_progress_bar")
con.register("nd", nodes)
con.register("ves", pd.read_parquet(paths.VESSELS)[
    ["mmsi","vessel_type","length"]])

print("=== 1. spells ===")
print(f"  draft is rescaled for 2023 onto the 2024-2026 scale "
      f"(x{draftfix.SCALE:.4f} about the median) -- see psld/draftfix.py")
DRAFT = draftfix.sql("draft", "t_min")
con.execute(f"""
CREATE OR REPLACE TABLE spells AS
WITH b AS (SELECT * FROM '{paths.WORK}/buckets/*.parquet'),
 f AS (SELECT *, CASE WHEN date_diff('minute', lag(t_max) OVER w, t_min) > {GAP_MIN}
         OR lag(t_max) OVER w IS NULL THEN 1 ELSE 0 END AS ns
       FROM b WINDOW w AS (PARTITION BY mmsi, node_id ORDER BY bucket)),
 g AS (SELECT *, sum(ns) OVER (PARTITION BY mmsi, node_id ORDER BY bucket) AS sid FROM f),
 s AS (SELECT mmsi, node_id, sid, min(t_min) AS t_in, max(t_max) AS t_out,
              sum(n) AS n_pings,
              -- the source changes draft scale at 2024-01-01; see psld/draftfix.py
              median({DRAFT}) AS draft
       FROM g GROUP BY mmsi, node_id, sid)
SELECT s.mmsi, s.node_id, nd.port, nd.complex, nd.node_kind, nd.region, nd.in_graph,
       s.t_in, s.t_out, date_diff('second', s.t_in, s.t_out)/3600.0 AS dur_h,
       s.n_pings, s.draft, v.length, v.vessel_type,
       (v.vessel_type BETWEEN 70 AND 79) AS is_cargo,
       s.t_in  <= TIMESTAMP '{SPAN0}' + INTERVAL {EDGE_H} HOUR AS left_cens,
       s.t_out >= TIMESTAMP '{SPAN1}' - INTERVAL {EDGE_H} HOUR AS right_cens
FROM s JOIN nd USING (node_id) LEFT JOIN ves v USING (mmsi)""")
sm = con.execute("""
SELECT node_kind, count(*) AS spells, count(DISTINCT mmsi) AS vessels,
  round(median(dur_h),2) AS median_h, round(quantile_cont(dur_h,0.9),1) AS p90_h,
  round(max(dur_h),1) AS max_h,
  count(*) FILTER (left_cens OR right_cens) AS censored,
  round(100.0*count(*) FILTER (left_cens OR right_cens)/count(*),2) AS pct_censored
FROM spells GROUP BY 1 ORDER BY 2 DESC""").df()
print(sm.to_string(index=False))

print("\n=== 2. port-call events (berth spells) ===")
con.execute(f"""
CREATE OR REPLACE TABLE ev_call AS
SELECT node_id, port, mmsi, t_in, t_out, dur_h, draft, length, is_cargo,
       left_cens, right_cens, NOT right_cens AS event_observed,
       CASE WHEN right_cens THEN 'window_end' ELSE 'departed' END AS exit_reason,
       dur_h > {LONG_STAY_D*24} AS long_stay
FROM spells WHERE node_kind = 'berth'""")
q = con.execute("""SELECT count(*) n, count(*) FILTER (long_stay) ls,
  round(median(dur_h),2) med, round(100.0*avg(event_observed::INT),2) obs FROM ev_call""").df()
print(f"  {int(q.n[0]):,} berth spells, median {q.med[0]:.1f} h, "
      f"{q.obs[0]:.1f}% ending inside the window")
print(f"  {int(q.ls[0]):,} run longer than {LONG_STAY_D} days (laid-up vessels, "
      f"flagged long_stay and excluded from the duration target)")

print("\n=== 3. berth-wait events (anchorage spells, with the competing risk) ===")
con.execute(f"""
CREATE OR REPLACE TABLE ev_wait AS
WITH a AS (SELECT * FROM spells WHERE node_kind = 'anchorage'),
     b AS (SELECT mmsi, complex, t_in FROM spells WHERE node_kind = 'berth')
SELECT a.node_id, a.port, a.mmsi, a.t_in, a.t_out, a.dur_h, a.draft, a.length,
       a.is_cargo, a.left_cens, a.right_cens,
       (SELECT min(b.t_in) FROM b WHERE b.mmsi = a.mmsi AND b.complex = a.complex
          AND b.t_in >= a.t_out AND b.t_in <= a.t_out + INTERVAL {BERTH_WITHIN_D} DAY)
       AS t_berth
FROM a""")
con.execute(f"""
CREATE OR REPLACE TABLE ev_wait AS
SELECT *, t_berth IS NOT NULL AS event_observed,
   CASE WHEN t_berth IS NOT NULL THEN 'berthed'
        WHEN right_cens THEN 'window_end' ELSE 'left' END AS exit_reason,
   CASE WHEN t_berth IS NOT NULL
        THEN date_diff('second', t_in, t_berth)/3600.0 END AS wait_to_berth_h
FROM ev_wait""")
q = con.execute("""SELECT count(*) n, count(*) FILTER (exit_reason='berthed') b,
  count(*) FILTER (exit_reason='left') l, count(*) FILTER (exit_reason='window_end') w,
  round(median(dur_h) FILTER (exit_reason='berthed'),2) med,
  round(median(wait_to_berth_h) FILTER (exit_reason='berthed'),2) med2 FROM ev_wait""").df()
print(f"  {int(q.n[0]):,} anchorage spells: {int(q.b[0]):,} berthed "
      f"({100*q.b[0]/q.n[0]:.0f}%), {int(q.l[0]):,} left without berthing "
      f"({100*q.l[0]/q.n[0]:.0f}%, competing risk), {int(q.w[0]):,} cut off by the window")
print(f"  median observed wait: {q.med[0]:.1f} h as anchorage-spell length, "
      f"{q.med2[0]:.1f} h measured to the first berth ping")

print("\n=== 4. events.parquet ===")
ev = pd.concat([
    con.execute("""SELECT node_id, port, mmsi, t_in AS entry_ts, 'port_call' AS target_kind,
        dur_h AS duration_h, event_observed, exit_reason, draft, length, is_cargo,
        left_cens, right_cens, long_stay, NULL::DOUBLE AS wait_to_berth_h
        FROM ev_call""").df(),
    con.execute("""SELECT node_id, port, mmsi, t_in AS entry_ts, 'berth_wait' AS target_kind,
        dur_h AS duration_h, event_observed, exit_reason, draft, length, is_cargo,
        left_cens, right_cens, false AS long_stay, wait_to_berth_h
        FROM ev_wait""").df()], ignore_index=True)
ev = ev.merge(nodes[["node_id","region","in_graph"]], on="node_id", how="left")
ev["entry_hour"] = ev.entry_ts.dt.floor("h")
ev = ev.sort_values(["target_kind","entry_ts"]).reset_index(drop=True)
ev.to_parquet(paths.WORK / "events.parquet", index=False)
con.execute(f"COPY spells TO '{paths.WORK}/spells.parquet' (COMPRESSION zstd)")
print(f"  events.parquet: {len(ev):,} rows "
      f"({int(ev.in_graph.sum()):,} in the East+Gulf graph)")
print(ev[ev.in_graph].groupby(["target_kind","exit_reason"]).size().rename("n").to_string())
print("\n  per-port, East+Gulf, observed events only:")
g = (ev[ev.in_graph & ev.event_observed & ~ev.long_stay]
     .pivot_table(index="port", columns="target_kind", values="duration_h",
                  aggfunc=["size","median"]).round(1))
g.columns = [f"{a}_{b}" for a, b in g.columns]
print(g.sort_values("size_port_call", ascending=False).to_string())
