"""Turn the staged per-day subsets into the final preprocessed dataset.

The per-day pass could only use that day's static fields, so it kept a loose
superset (Class A and (cargo/tanker type OR length >= 100 m)).  Here the vessel
attributes are resolved once across the whole period, the strict filter is
applied, and the isolated-teleport filter runs over each vessel's complete
series rather than one day at a time.
"""
import duckdb, os, glob, time, shutil
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from psld import paths, config
PINGS, VESSELS = str(paths.PINGS), str(paths.VESSELS)
DERIVED, STAGED = str(paths.DERIVED), str(paths.STAGED)
t0 = time.time()
os.makedirs(paths.PREPROCESSED, exist_ok=True)
# COPY ... PARTITION_BY with OVERWRITE_OR_IGNORE writes into existing directories
# without clearing them.  Re-running with different parallelism then leaves
# orphaned data_*.parquet files from the previous run, which read back as extra
# rows.  Clear the target first so the step is idempotent.
if os.path.isdir(PINGS):
    shutil.rmtree(PINGS)          # this script's own previous output
    print(f"cleared existing {PINGS}")
con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB'")
os.makedirs(paths.DUCKDB_TMP, exist_ok=True)
con.execute(f"PRAGMA temp_directory='{paths.DUCKDB_TMP}'")

con.execute(f"CREATE VIEW vd AS SELECT * FROM '{DERIVED}/vessel_day/*.parquet'")
con.execute(f"CREATE VIEW sp AS SELECT * FROM '{STAGED}/day=*/*.parquet'")
print("staged pings:", f"{con.execute('SELECT count(*) FROM sp').fetchone()[0]:,}")
print("staged days:  ", con.execute("SELECT count(DISTINCT day) FROM vd").fetchone()[0])

# ---- 1. global vessel attributes -------------------------------------------
con.execute("""
CREATE OR REPLACE TABLE vessels_all AS
SELECT mmsi,
       arg_max(transceiver, pings)  AS transceiver,   -- the value on its busiest day
       arg_max(vessel_type, pings)  AS vessel_type,
       max(length)                  AS length,
       max(width)                   AS width,
       arg_max(vessel_name, pings)  AS vessel_name,
       sum(pings)                   AS pings_raw,
       count(*)                     AS days_seen,
       min(day)                     AS first_day,
       max(day)                     AS last_day
FROM vd GROUP BY mmsi""")
tot = con.execute("SELECT count(*) FROM vessels_all").fetchone()[0]
con.execute("""
CREATE OR REPLACE TABLE vessels AS
SELECT * FROM vessels_all
WHERE transceiver='A' AND vessel_type BETWEEN 70 AND 89 AND length >= 100""")
kept = con.execute("SELECT count(*) FROM vessels").fetchone()[0]
print(f"vessels: {tot:,} seen -> {kept:,} deep-sea Class A (type 70-89, length >= 100 m)")

# ---- 2. how much did the per-day loose filter cost? ------------------------
exp = con.execute("""SELECT sum(v.pings) FROM vd v SEMI JOIN vessels k ON v.mmsi=k.mmsi""").fetchone()[0]
got = con.execute("""SELECT count(*) FROM sp s SEMI JOIN vessels k ON s.mmsi=k.mmsi""").fetchone()[0]
print(f"pings of qualifying vessels: {exp:,} broadcast, {got:,} staged "
      f"({100*got/exp:.2f}% retained; the rest are rows where that day's static "
      f"fields did not yet identify the vessel, plus out-of-domain positions)")

# ---- 3. teleport filter over each vessel's full series ---------------------
# The window pass is split into buckets on mmsi so memory stays bounded as the
# archive grows (a single window over ~1B staged rows spills heavily).  mmsi
# never straddles a bucket, so bucketing cannot change the result.
N_BUCKETS = 8
con.execute("CREATE OR REPLACE TABLE bad (mmsi BIGINT, ts TIMESTAMP)")
for b in range(N_BUCKETS):
    con.execute(f"""
INSERT INTO bad
WITH s AS (
  SELECT p.mmsi, p.ts, p.lon, p.lat,
     lag(p.lon)  OVER w AS plon, lag(p.lat)  OVER w AS plat,
     lead(p.lon) OVER w AS nlon, lead(p.lat) OVER w AS nlat,
     date_diff('second', lag(p.ts) OVER w, p.ts)  AS dtp,
     date_diff('second', p.ts, lead(p.ts) OVER w) AS dtn
  FROM sp p SEMI JOIN vessels k ON p.mmsi = k.mmsi
  WHERE p.mmsi % {N_BUCKETS} = {b}
  WINDOW w AS (PARTITION BY p.mmsi ORDER BY p.ts))
SELECT mmsi, ts FROM s
WHERE 2*6371*asin(sqrt(pow(sin(radians(lat-plat)/2),2)+cos(radians(plat))
        *cos(radians(lat))*pow(sin(radians(lon-plon)/2),2)))/(dtp/3600.0)/1.852 > 60
  AND 2*6371*asin(sqrt(pow(sin(radians(nlat-lat)/2),2)+cos(radians(lat))
        *cos(radians(nlat))*pow(sin(radians(nlon-lon)/2),2)))/(dtn/3600.0)/1.852 > 60""")
    print(f"  teleport scan bucket {b+1}/{N_BUCKETS}", flush=True)
nbad = con.execute("SELECT count(*) FROM bad").fetchone()[0]
print(f"isolated teleports dropped: {nbad:,} ({100*nbad/got:.4f}%)")

# ---- 4. write final ---------------------------------------------------------
years = [r[0] for r in con.execute(
    "SELECT DISTINCT year(ts) AS y FROM sp ORDER BY y").fetchall()]
for yr in years:
    con.execute(f"""
    COPY (SELECT p.mmsi, p.ts, p.lon, p.lat, p.sog, p.status, p.draft, p.h3_10,
                 -- recomputed here, not inherited: `sog < 1 OR status IN (1,5)`
                 -- yields NULL when both inputs are NULL, and a nullable flag
                 -- silently behaves as false in every downstream WHERE clause
                 coalesce(p.sog < 1, false) OR coalesce(p.status IN (1,5), false)
                   AS stationary,
                 p.ts::DATE AS day
          FROM sp p SEMI JOIN vessels k ON p.mmsi = k.mmsi
          ANTI JOIN bad b ON p.mmsi = b.mmsi AND p.ts = b.ts
          WHERE year(p.ts) = {yr})
      TO '{PINGS}' (FORMAT PARQUET, PARTITION_BY (day),
          COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1)""")
    print(f"  wrote {yr}", flush=True)
con.execute(f"COPY (SELECT * FROM vessels) TO '{VESSELS}' (FORMAT PARQUET)")
con.execute(f"COPY (SELECT * FROM vessels_all) TO '{DERIVED}/vessels_all.parquet' (FORMAT PARQUET)")

n = con.execute(f"SELECT count(*) FROM '{PINGS}/**/*.parquet'").fetchone()[0]
size = sum(os.path.getsize(f) for f in glob.glob(f"{paths.PREPROCESSED}/**/*.parquet", recursive=True))
days = con.execute(f"SELECT count(DISTINCT day) FROM '{PINGS}/**/*.parquet'").fetchone()[0]
print(f"\nfinal: {n:,} pings over {days} days, {kept:,} vessels, {size/1e9:.2f} GB")
print(f"elapsed {time.time()-t0:.0f}s")
