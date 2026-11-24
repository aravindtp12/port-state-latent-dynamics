"""Download the Natural Earth 10 m shapefiles used for land / coastline / lake geometry.

    python pipeline/00_download_geo.py

Used to classify hexagons as land, shore or open water when the berth and anchorage regions
are built (psld/geoutil.py) and to draw the maps.  Natural Earth is public domain.
Writes data/geo/ne_10m_{land,coastline,lakes}.*

The files served are byte-identical to the ones used for the paper (land 5.1.1,
coastline 5.0.0-pre9, lakes 5.0.0 per their VERSION.txt).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import io, urllib.request, zipfile

from psld import paths

BASE = "https://naturalearth.s3.amazonaws.com/10m_physical"
paths.ensure(paths.GEO)
for name in ("ne_10m_land", "ne_10m_coastline", "ne_10m_lakes"):
    if (paths.GEO / f"{name}.shp").exists():
        print(f"{name}: already present"); continue
    print(f"{name}: downloading ...", flush=True)
    with urllib.request.urlopen(f"{BASE}/{name}.zip", timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(paths.GEO)
print("done ->", paths.GEO)
