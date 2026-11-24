"""Fetch and preprocess all published 2026 AIS data (2026-01-01 .. 2026-06-30).

Raw CSVs are streamed: download -> preprocess -> delete, three downloads in
flight while one day is processed, so peak disk use stays near 1.5 GB even
though the compressed archive is 41 GB.

Per day it writes three things:
  data/staged_pings/day=D/    loose deep-sea ping subset (final filter applied in step 2)
  data/derived/vessel_day/D   per-vessel summary for ALL vessels that day
  data/derived/hex7_day/D     H3 res-7 counts for ALL traffic that day
The last two are the hedge against deleting the raw files: fleet-level and
coverage-level questions stay answerable without re-downloading 41 GB.
"""
import duckdb, os, sys, time, threading, queue, urllib.request, urllib.error, datetime, glob
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from psld import paths, config

# usage: python pipeline/01_fetch_ais.py [year] [n_parallel]
YEAR  = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
N_DL  = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE = f"https://noaaocm.blob.core.windows.net/ais/csv2/csv{YEAR}"
STAGE, OUT = str(paths.STAGING), str(paths.STAGED)
DERIVED = str(paths.DERIVED)
DOM = (-180, 5, -60, 75)
READ_TIMEOUT = 60      # seconds without data before a transfer is abandoned
# Measured aggregate throughput against this host: 3.8 MB/s at 1 connection,
# 10.4 at 8, 11.4 at 12 -- it flattens past 8, so 8 is the useful setting.
for d in (STAGE, OUT, f"{DERIVED}/vessel_day", f"{DERIVED}/hex7_day"):
    os.makedirs(d, exist_ok=True)

start, end = datetime.date(YEAR,1,1), datetime.date(YEAR,12,31)
days = [(start + datetime.timedelta(i)).isoformat() for i in range((end-start).days + 1)]
# the study period ends 2026-06-30 (the archive publishes nothing for H2 2026)
days = [d for d in days if config.SPAN0 <= d <= config.SPAN1]
done = {os.path.basename(p).split("=")[1] for p in glob.glob(f"{OUT}/day=*")}
todo = [d for d in days if d not in done]
print(f"{YEAR}: {len(days)} days in range, {len(done)} already processed, {len(todo)} to go", flush=True)

def local_source(day):
    """Prefer an already-downloaded raw file over re-fetching it."""
    for p in (f"{STAGE}/ais-{day}.csv", f"{STAGE}/ais-{day}.csv.zst"):
        if os.path.exists(p):
            return p
    return None

def download(day):
    """Fetch one day, resuming a partial file if present.

    urlretrieve() was used here originally and hung the whole run: it applies no
    socket timeout, so when a TCP transfer stalls mid-body the read blocks
    forever and the retry loop below never fires (a hang raises nothing).  Every
    read now runs under an explicit timeout, and because the server answers
    ranged requests with 206 a stalled attempt resumes instead of restarting.
    """
    p = local_source(day)
    if p:
        return p
    dst = f"{STAGE}/ais-{day}.csv.zst"
    tmp = dst + ".part"
    url = f"{BASE}/ais-{day}.csv.zst"
    for attempt in range(6):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
                resumed = (r.status == 206)
                total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
                if not resumed:
                    have = 0
                with open(tmp, "ab" if resumed else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)          # raises on a stalled socket
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
            if total and have != total:
                raise IOError(f"short read {have}/{total}")
            os.replace(tmp, dst)
            return dst
        except urllib.error.HTTPError as e:
            if e.code == 404:                   # not published: retrying cannot help
                print(f"  !! {day}: not published (404)", flush=True)
                return None
            print(f"  [dl retry {attempt+1}/6] {day}: HTTP {e.code}", flush=True)
            time.sleep(min(5 * (attempt + 1), 30))
        except Exception as e:
            print(f"  [dl retry {attempt+1}/6] {day}: {type(e).__name__}: {e}", flush=True)
            time.sleep(min(5 * (attempt + 1), 30))
    print(f"  !! giving up on {day}", flush=True)
    return None

# bounded queue: at most N_DL+1 raw files on disk at once
q = queue.Queue(maxsize=N_DL + 1)
def worker(sub):
    for day in sub:
        q.put((day, download(day)))
threads = [threading.Thread(target=worker, args=(todo[i::N_DL],), daemon=True) for i in range(N_DL)]
for t in threads: t.start()

con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB'")
con.execute("INSTALL h3 FROM community; LOAD h3;")

COLS = ("{'mmsi':'BIGINT','base_date_time':'TIMESTAMP','longitude':'DOUBLE','latitude':'DOUBLE',"
        "'sog':'DOUBLE','cog':'DOUBLE','heading':'DOUBLE','vessel_name':'VARCHAR','imo':'VARCHAR',"
        "'call_sign':'VARCHAR','vessel_type':'INTEGER','status':'INTEGER','length':'DOUBLE',"
        "'width':'DOUBLE','draft':'DOUBLE','cargo':'INTEGER','transceiver':'VARCHAR'}")

def process(day, path):
    comp = "zstd" if path.endswith(".zst") else "none"
    src = f"read_csv('{path}', header=true, columns={COLS}, compression='{comp}')"
    con.execute(f"CREATE OR REPLACE TEMP VIEW r AS SELECT * FROM {src}")
    con.execute(f"""COPY (
      SELECT mmsi, mode(transceiver) AS transceiver, mode(vessel_type) AS vessel_type,
             max(length) AS length, max(width) AS width, any_value(vessel_name) AS vessel_name,
             count(*) AS pings, DATE '{day}' AS day
      FROM r GROUP BY mmsi)
      TO '{DERIVED}/vessel_day/{day}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.execute(f"""COPY (
      SELECT h3_latlng_to_cell(latitude, longitude, 7) AS h3_7, count(*) AS n,
             count(*) FILTER (sog < 1 OR status IN (1,5)) AS n_stationary,
             count(*) FILTER (transceiver='A') AS n_class_a,
             count(DISTINCT mmsi) AS vessels, DATE '{day}' AS day
      FROM r WHERE longitude BETWEEN {DOM[0]} AND {DOM[2]}
               AND latitude BETWEEN {DOM[1]} AND {DOM[3]}
      GROUP BY 1)
      TO '{DERIVED}/hex7_day/{day}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    os.makedirs(f"{OUT}/day={day}", exist_ok=True)
    n = con.execute(f"""COPY (
      SELECT DISTINCT ON (mmsi, base_date_time)
             mmsi, base_date_time AS ts, longitude AS lon, latitude AS lat,
             sog, status, draft,
             h3_latlng_to_cell(latitude, longitude, 10) AS h3_10,
             (sog < 1 OR status IN (1,5)) AS stationary
      FROM r
      WHERE transceiver='A' AND (vessel_type BETWEEN 70 AND 89 OR length >= 100)
        AND longitude BETWEEN {DOM[0]} AND {DOM[2]}
        AND latitude BETWEEN {DOM[1]} AND {DOM[3]}
      ORDER BY mmsi, base_date_time)
      TO '{OUT}/day={day}/part.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    return con.execute(f"SELECT count(*) FROM '{OUT}/day={day}/part.parquet'").fetchone()[0]

t0 = time.time(); ok = 0; failed = []
for i in range(len(todo)):
    day, path = q.get()
    if path is None:
        print(f"  !! download failed permanently: {day}", flush=True); failed.append(day); continue
    try:
        n = process(day, path)
        ok += 1
        if path.startswith(STAGE):
            os.remove(path)                      # raw file deleted immediately
        el = time.time() - t0
        print(f"[{ok}/{len(todo)}] {day}: {n:,} pings kept | {el/60:.1f} min elapsed "
              f"| eta {(el/ok)*(len(todo)-ok)/60:.0f} min", flush=True)
    except Exception as e:
        print(f"  !! process failed {day}: {e}", flush=True); failed.append(day)

print(f"\ndone: {ok} days, {len(failed)} failed {failed}", flush=True)
