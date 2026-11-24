"""The evaluation protocol shared by every model, and the scoring functions.

Forecasts are made from rolling origins every 24 h through the test quarter, with a
72 h horizon.  The forecast for hour t+k is scored against hour t+k, per channel,
pooled over ports and origins, and only where the target is observed
(`target_mask`).  Occupancy is scored in vessels and the duration channels in hours;
channels are never pooled.
"""
import numpy as np

from psld import paths

CHANNELS = ["anchorage_occ", "berth_occ", "avg_berth_wait", "avg_port_call"]
UNITS = {"anchorage_occ": "vessels", "berth_occ": "vessels",
         "avg_berth_wait": "hours", "avg_port_call": "hours"}
H = 72                                  # forecast horizon, hours
STRIDE = 24                             # hours between forecast origins
FIRST_OFFSET = 48                       # first origin is 48 h into the test split
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99)
QS = np.array(QUANTILES)
MEDIAN = QUANTILES.index(0.50)


def test_origins(d):
    """Forecast origins (hour indices) used by every model: 87 of them."""
    te0, te1 = d["split"]["test"]
    return list(range(te0 + FIRST_OFFSET, te1 - H + 1, STRIDE))


def series(d):
    """The forecast targets as flat series j = (port, channel).

    Returns names, channel index per series, values S [T,J], validity V [T,J]."""
    tg = d["targets"]
    names = [f"{t['port']}|{CHANNELS[t['channel']]}" for t in tg]
    chan = np.array([t["channel"] for t in tg])
    S = np.array([d["Y"][:, t["node_index"], t["channel"]] for t in tg]).T.astype(float)
    V = np.array([d["target_mask"][:, t["node_index"], t["channel"]] for t in tg]).T
    return names, chan, S, V


def truth(d, origins):
    """TRUE and VALID, both [O,H,J]: observed values and their validity mask."""
    _, _, S, V = series(d)
    return (np.stack([S[o:o + H] for o in origins]),
            np.stack([V[o:o + H] for o in origins]))


# ------------------------------------------------------------------- scoring
def mae_at(P, TRUE, VALID, chan, c, k, sel=None):
    """MAE of the forecast for hour t+k, channel c, pooled over series and origins."""
    s = (chan == c) if sel is None else ((chan == c) & sel)
    e = np.abs(P[:, k - 1, :] - TRUE[:, k - 1, :])[:, s]
    v = VALID[:, k - 1, :][:, s]
    ok = v & np.isfinite(e)
    return float(e[ok].mean())


def crps_from_quantiles(Q, y):
    """CRPS = 2 * integral_0^1 pinball_a(y, q_a) da.

    The quantile grid is deliberately tail-dense, so a plain mean over the grid is
    not that integral.  Integrate by trapezoid in a, extending flat to a=0 and a=1
    so the outer mass is included.  Q [..., 9], y [...] -> [...]."""
    e = y[..., None] - Q
    pin = np.maximum(QS * e, (QS - 1.0) * e)
    a = np.concatenate([[0.0], QS, [1.0]])
    pin = np.concatenate([pin[..., :1], pin, pin[..., -1:]], -1)
    return 2.0 * np.trapezoid(pin, a, axis=-1)


def crps_at(Q, TRUE, VALID, chan, c, k, sel=None):
    s = (chan == c) if sel is None else ((chan == c) & sel)
    y = TRUE[:, k - 1, :][:, s]
    q = Q[:, k - 1, :, :][:, s]
    v = VALID[:, k - 1, :][:, s]
    return float(crps_from_quantiles(q, y)[v].mean())


def coverage90(Q, TRUE, VALID, chan, c):
    """Share of observed outcomes inside the central 90% interval [q05, q95]."""
    s = chan == c
    y, v, q = TRUE[..., s], VALID[..., s], Q[..., s, :]
    return float(100 * (((y >= q[..., 0]) & (y <= q[..., 6]))[v]).mean())


def save_array(name, arr):
    paths.ensure(paths.RESULTS)
    np.save(paths.RESULTS / name, arr)
    print(f"  -> {paths.RESULTS / name}  {arr.shape}")
