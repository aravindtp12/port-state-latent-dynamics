"""Classical baselines: seasonal-naive, hour-of-week climatology, VAR and SARIMA.

    python models/run_baselines.py                          # East+Gulf, 44 series
    SCOPE=all python models/run_baselines.py --domain west  # fitted on the West-coast series

Every baseline forecasts each of the four channels directly as an hourly series, on the
same 72 h horizon and the same rolling test origins as the neural models.

  seasonal-naive  y(t+k) = y(t+k-168)
  climatology     hour-of-week mean over the training period, per series
  VAR             joint over the series, per-series standardised, order by AIC (<= 24 lags),
                  fitted on the training period
  SARIMA          one per series, fitted on the last 365 days of training with seasonal period
                  24 and its parameters then held fixed while the state is filtered through the
                  history up to each origin.  East: order chosen by AIC from
                  {(2,0,1)(1,1,1)_24, (1,0,1)(1,1,1)_24}; West: (2,0,1)(1,1,1)_24.

Hours the feed was down, and duration hours whose trailing window held no completed stay,
are interpolated on the INPUT side and excluded from scoring through the validity mask.

Writes data/results/[west_]classical_predictions.npz (arrays [origins, 72 h, series]).
SARIMA dominates the run time (44 fits plus 87 filtered forecasts each).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, ast, time, warnings
import numpy as np
import pandas as pd

from psld import paths, config, data, protocol

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--domain", choices=["east", "west"], default="east")
args = ap.parse_args()
WEST = args.domain == "west"
assert WEST == (config.SCOPE == "all"), "--domain west needs SCOPE=all (east needs the default scope)"

H = protocol.H
SARIMA_GRID = [((2, 0, 1), (1, 1, 1, 24)), ((1, 0, 1), (1, 1, 1, 24))]
SARIMA_FIT_H = 365 * 24
MAX_VAR_LAGS = 24

d = data.load()
names, chan, S, V = protocol.series(d)
J = S.shape[1]
tr0, tr1 = d["split"]["train"]
origins = protocol.test_origins(d)
west_set = set(data.west_nodes(d)) if WEST else None
fit_j = [j for j, t in enumerate(d["targets"]) if (not WEST) or t["node_index"] in west_set]
print(f"{args.domain}: {len(fit_j)} of {J} series fitted, {len(origins)} test origins")

Sf = pd.DataFrame(S).mask(~V).interpolate().bfill().ffill().values        # input side only
hours = pd.DatetimeIndex(d["hours"])
out = {}

# ---- seasonal-naive and hour-of-week climatology
out["seasonal_naive"] = np.stack([Sf[o - 168:o - 168 + H] for o in origins])
hw = hours[tr0:tr1]
clim = pd.DataFrame(np.where(V[tr0:tr1], S[tr0:tr1], np.nan)).groupby([hw.dayofweek, hw.hour]).mean()
climfull = clim.reindex(pd.MultiIndex.from_arrays([hours.dayofweek, hours.hour])).values
out["climatology"] = np.stack([climfull[o:o + H] for o in origins])

# ---- VAR
from statsmodels.tsa.api import VAR
t0 = time.time()
mu, sd = Sf[tr0:tr1][:, fit_j].mean(0), Sf[tr0:tr1][:, fit_j].std(0) + 1e-9
Z = (Sf[:, fit_j] - mu) / sd
vm = VAR(Z[tr0:tr1]); p = int(vm.select_order(maxlags=MAX_VAR_LAGS).aic); vr = vm.fit(p)
fc = np.stack([vr.forecast(Z[o - p:o], steps=H) for o in origins]) * sd + mu
P = np.full((len(origins), H, J), np.nan, np.float32)
P[:, :, fit_j] = fc
out["VAR"] = P
print(f"VAR: order {p} by AIC, {vr.params.size:,} coefficients, {time.time() - t0:.0f}s")

# ---- SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
orders_csv = paths.RESULTS / f"{'west_' if WEST else ''}sarima_orders.csv"
prior = {}
if orders_csv.exists():
    prior = {r.series: (ast.literal_eval(r.order), ast.literal_eval(r.seasonal))
             for r in pd.read_csv(orders_csv).itertuples()}
    print(f"  reusing {len(prior)} cached SARIMA order selections")


def fit(y, order, seasonal, simple):
    """simple_differencing=True is faster but makes predicted_mean the forecast of the
    DIFFERENCED series -- used for AIC ranking only, never to forecast."""
    return SARIMAX(y, order=order, seasonal_order=seasonal, simple_differencing=simple,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=0, maxiter=50)


Psar = np.full((len(origins), H, J), np.nan, np.float32)
chosen, t0 = [], time.time()
for n_, j in enumerate(fit_j):
    y0 = max(tr0, tr1 - SARIMA_FIT_H)
    ytr = Sf[y0:tr1, j]
    if names[j] in prior:
        od, sdd = prior[names[j]]
    elif WEST:
        od, sdd = SARIMA_GRID[0]
    else:
        _, od, sdd = min(((fit(ytr, o_, s_, True).aic, o_, s_) for o_, s_ in SARIMA_GRID), key=lambda x: x[0])
    res = fit(ytr, od, sdd, False)
    chosen.append(dict(series=names[j], order=str(od), seasonal=str(sdd), aic=round(res.aic)))
    for i, o in enumerate(origins):
        Psar[i, :, j] = res.apply(Sf[y0:o, j]).get_forecast(H).predicted_mean
    print(f"  SARIMA [{n_ + 1:2d}/{len(fit_j)}] {names[j]:38s} {od}{sdd}  [{time.time() - t0:5.0f}s]", flush=True)
out["SARIMA"] = Psar
pd.DataFrame(chosen).to_csv(orders_csv, index=False)

paths.ensure(paths.RESULTS)
fname = paths.RESULTS / f"{'west_' if WEST else ''}classical_predictions.npz"
np.savez_compressed(fname, **out)
print(f"-> {fname}  {sorted(out)}")
