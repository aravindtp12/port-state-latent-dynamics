"""Coastline-distance and land-mask helpers.

Distances are computed in a locally-projected frame (lon scaled by cos(lat) of a
5-degree latitude band, both axes in km) so shapely's planar nearest-segment
search returns true kilometres rather than degrees.
"""
import numpy as np, pyogrio, pyogrio.raw, shapely
from shapely import STRtree
from shapely.ops import transform

from psld import paths
GEO = paths.GEO
KM_PER_DEG_LAT = 110.574

def _read(layer, bbox=None, columns=None):
    """-> (fields dict, shapely geometry array). Avoids the geopandas dependency."""
    meta, _fids, wkb, fields = pyogrio.raw.read(
        str(GEO / f"{layer}.shp"), bbox=bbox, columns=columns)
    geoms = shapely.from_wkb(wkb)
    return dict(zip(meta["fields"], fields)), geoms

def _scale(geoms, coslat):
    return shapely.transform(
        geoms, lambda c: np.column_stack([c[:, 0] * 111.320 * coslat,
                                          c[:, 1] * KM_PER_DEG_LAT]))

def coast_distance_km(lon, lat, band=5.0, verbose=False):
    """Distance (km) from each point to the nearest coastline.

    Works in 5-degree latitude bands, each projected to km (lon scaled by
    cos(band centre)), so shapely's planar nearest-segment search returns
    kilometres.  The coastline window starts at +/-8 degrees and is widened for
    any point whose answer is too close to the window edge to be trustworthy.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    out = np.full(lon.shape, np.nan)
    bidx = np.floor(lat / band).astype(int)
    for b in np.unique(bidx):
        lo, hi = b * band, (b + 1) * band
        in_band = np.flatnonzero(bidx == b)
        latc = (lo + hi) / 2
        coslat = np.cos(np.radians(latc))
        todo = in_band
        for margin in (8.0, 25.0, 70.0, 180.0):
            if len(todo) == 0:
                break
            _f, coast = _read("ne_10m_coastline", columns=[],
                              bbox=(max(lon[todo].min() - margin, -180), lo - margin,
                                    min(lon[todo].max() + margin, 180), hi + margin))
            if len(coast) == 0:
                continue
            g = _scale(coast, coslat)
            tree = STRtree(g)
            pts = shapely.points(lon[todo] * 111.320 * coslat, lat[todo] * KM_PER_DEG_LAT)
            pair, dist = tree.query_nearest(pts, all_matches=False, return_distance=True)
            d = np.full(len(pts), np.inf)
            d[pair[0]] = dist
            out[todo] = d
            if verbose:
                print(f"  band {lo:+.0f}..{hi:+.0f} margin {margin:.0f}deg: "
                      f"{len(todo):,} pts, {len(coast)} coast parts", flush=True)
            # an answer within 60% of the window half-width is safely interior
            trust_km = 0.6 * margin * 111.320 * coslat
            todo = todo[d > trust_km]
    return out

_LAND = {}
def land_tree():
    if "t" not in _LAND:
        _f, g = _read("ne_10m_land", columns=[])
        shapely.prepare(g)
        _LAND["g"] = g; _LAND["t"] = STRtree(g)
    return _LAND["t"], _LAND["g"]

def on_land(lon, lat):
    """True where the point falls inside a Natural Earth land polygon."""
    tree, g = land_tree()
    pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
    res = np.zeros(len(pts), bool)
    pairs = tree.query(pts, predicate="intersects")
    res[pairs[0]] = True
    return res

def ports():
    """-> (fields dict, point geometries) for Natural Earth ports."""
    return _read("ne_10m_ports")


# ---- shore distance that also knows about lakes (Great Lakes, inland water) ----
def _shore_geoms(bbox, coslat):
    """Coastline + lake shorelines, scaled to km, as one array."""
    parts = []
    _f, coast = _read("ne_10m_coastline", columns=[], bbox=bbox)
    if len(coast):
        parts.append(_scale(coast, coslat))
    _f, lakes = _read("ne_10m_lakes", columns=[], bbox=bbox)
    if len(lakes):
        parts.append(_scale(shapely.boundary(lakes), coslat))
    return np.concatenate(parts) if parts else np.array([], dtype=object)

def shore_distance_km(lon, lat, band=5.0, verbose=False):
    """Distance (km) to the nearest shoreline, counting lake shores as shore."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    out = np.full(lon.shape, np.nan)
    bidx = np.floor(lat / band).astype(int)
    for b in np.unique(bidx):
        lo, hi = b * band, (b + 1) * band
        in_band = np.flatnonzero(bidx == b)
        coslat = np.cos(np.radians((lo + hi) / 2))
        todo = in_band
        for margin in (8.0, 25.0, 70.0, 180.0):
            if len(todo) == 0:
                break
            g = _shore_geoms((max(lon[todo].min() - margin, -180), lo - margin,
                              min(lon[todo].max() + margin, 180), hi + margin), coslat)
            if len(g) == 0:
                continue
            tree = STRtree(g)
            pts = shapely.points(lon[todo] * 111.320 * coslat, lat[todo] * KM_PER_DEG_LAT)
            pair, dist = tree.query_nearest(pts, all_matches=False, return_distance=True)
            d = np.full(len(pts), np.inf); d[pair[0]] = dist
            out[todo] = d
            if verbose:
                print(f"  band {lo:+.0f}..{hi:+.0f} margin {margin:.0f}: {len(todo):,} pts,"
                      f" {len(g)} shore parts", flush=True)
            todo = todo[d > 0.6 * margin * 111.320 * coslat]
    return out

_LAKE = {}
def in_lake(lon, lat):
    """True where the point falls inside a Natural Earth lake polygon."""
    if "t" not in _LAKE:
        _f, g = _read("ne_10m_lakes", columns=[])
        shapely.prepare(g)
        _LAKE["g"] = g; _LAKE["t"] = STRtree(g)
    pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
    res = np.zeros(len(pts), bool)
    pairs = _LAKE["t"].query(pts, predicate="intersects")
    res[pairs[0]] = True
    return res

def classify(lon, lat):
    """-> ('sea'|'lake'|'inland', signed shore distance km: + water, - inside land)."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    land = on_land(lon, lat); lake = in_lake(lon, lat)
    kind = np.where(~land, "sea", np.where(lake, "lake", "inland")).astype(object)
    d = shore_distance_km(lon, lat)
    signed = np.where(kind == "inland", -d, d)
    return kind, signed
