"""Score every East+Gulf model on the test origins: Table I (MAE) and Table II (CRPS).

    python models/score_east.py

Reads whatever the training scripts have written to data/results/ and skips models that
are missing.  MAE is of the point forecast (the median for the quantile models) for the
forecast at hour t+k, k = 12 and 72, per channel, pooled over the 11 ports and all test
origins.  CRPS is evaluated for the models with a nine-quantile head (a point forecast's
CRPS is its MAE).  Tables are printed and written to data/results/table_*.csv.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from psld import paths, data, protocol as P

KS = (12, 72)
d = data.load()
origins = P.test_origins(d)
TRUE, VALID = P.truth(d, origins)
_, chan, _, _ = P.series(d)
R = paths.RESULTS


def load(name):
    f = R / name
    return np.load(f) if f.exists() else None


point, quant = {}, {}
cl = R / "classical_predictions.npz"
if cl.exists():
    z = np.load(cl)
    for k, lab in [("seasonal_naive", "Seasonal-naive"), ("climatology", "Climatology"),
                   ("VAR", "VAR"), ("SARIMA", "SARIMA")]:
        point[lab] = z[k]
for lab, f in [("Graph WaveNet", "gwnet_point_predictions.npy"), ("DCRNN", "dcrnn_point_predictions.npy")]:
    if (a := load(f)) is not None:
        point[lab] = a
for lab, f in [("PSLD", "psld_quantiles.npy"),
               ("Graph WaveNet + quantile head", "gwnet_quantile_quantiles.npy"),
               ("DCRNN + quantile head", "dcrnn_quantile_quantiles.npy")]:
    if (a := load(f)) is not None:
        quant[lab] = a
if "PSLD" in quant:
    point["PSLD"] = quant["PSLD"][..., P.MEDIAN]

rows = [dict(model=m, channel=P.CHANNELS[c], horizon_h=k, mae=P.mae_at(pr, TRUE, VALID, chan, c, k))
        for m, pr in point.items() for c in range(4) for k in KS]
mae = pd.DataFrame(rows)
tab1 = mae.pivot_table(index="model", columns=["channel", "horizon_h"], values="mae")
tab1 = tab1[[(P.CHANNELS[c], k) for c in range(4) for k in KS]].reindex(list(point))
pd.set_option("display.width", 200)
print("Table I -- test MAE at 12 h and 72 h (occupancy in vessels, durations in hours)")
print(tab1.round(2).to_string())
tab1.to_csv(R / "table_mae.csv")

if quant:
    rows = []
    for m, q in quant.items():
        for c in range(4):
            r = dict(model=m, channel=P.CHANNELS[c], cov90=P.coverage90(q, TRUE, VALID, chan, c))
            for k in KS:
                r[f"crps_{k}h"] = P.crps_at(q, TRUE, VALID, chan, c, k)
            rows.append(r)
    crps = pd.DataFrame(rows)
    tab2 = crps.pivot_table(index="model", columns="channel", values=[f"crps_{k}h" for k in KS])
    tab2 = tab2[[(f"crps_{k}h", P.CHANNELS[c]) for k in KS for c in range(4)]]
    print("\nTable II -- test CRPS at 12 h and 72 h (every model with the same nine-quantile head)")
    print(tab2.round(2).to_string())
    print("\n90% interval coverage (nominal 90)")
    print(crps.pivot_table(index="model", columns="channel", values="cov90").round(1).to_string())
    crps.to_csv(R / "table_crps.csv", index=False)
