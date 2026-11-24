"""Score the West-coast transfer study: Table III.

    SCOPE=all python models/score_west.py

Test MAE at 12 h and 72 h pooled over Long Beach, Oakland and Vancouver, per channel
(channels are never pooled: two are vessel counts, two are durations in hours).
Compares the classical models and the graph networks fitted directly on West-coast
data with PSLD applied zero-shot and fine-tuned.  Also prints CRPS and 90% coverage
for the two PSLD variants.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from psld import paths, config, data, protocol as P

assert config.SCOPE == "all", "run with SCOPE=all"
KS = (12, 72)
d = data.load()
origins = P.test_origins(d)
TRUE, VALID = P.truth(d, origins)
_, chan, _, _ = P.series(d)
west_nodes = set(data.west_nodes(d))
west = np.array([t["node_index"] in west_nodes for t in d["targets"]])
R = paths.RESULTS

point, quant = {}, {}
if (R / "west_classical_predictions.npz").exists():
    z = np.load(R / "west_classical_predictions.npz")
    for k, lab in [("seasonal_naive", "Seasonal-naive"), ("VAR", "VAR"), ("SARIMA", "SARIMA")]:
        point[lab] = z[k]
for lab, f in [("Graph WaveNet", "west_gwnet_point_predictions.npy"), ("DCRNN", "west_dcrnn_point_predictions.npy")]:
    if (R / f).exists():
        point[lab] = np.load(R / f)
for lab, f in [("PSLD (zero shot)", "west_psld_zeroshot_quantiles.npy"),
               ("PSLD (fine-tuned)", "west_psld_finetuned_quantiles.npy")]:
    if (R / f).exists():
        quant[lab] = np.load(R / f)
        point[lab] = quant[lab][..., P.MEDIAN]

rows = [dict(model=m, channel=P.CHANNELS[c], horizon_h=k, mae=P.mae_at(pr, TRUE, VALID, chan, c, k, west))
        for m, pr in point.items() for c in range(4) for k in KS]
tab = pd.DataFrame(rows).pivot_table(index="model", columns=["channel", "horizon_h"], values="mae")
tab = tab[[(P.CHANNELS[c], k) for c in range(4) for k in KS]].reindex(list(point))
pd.set_option("display.width", 200)
print("Table III -- West-coast test MAE at 12 h and 72 h, pooled over Long Beach, Oakland, Vancouver")
print(tab.round(2).to_string())
tab.to_csv(R / "table_west_mae.csv")

for m, q in quant.items():
    print(f"\n{m}: CRPS (12 h / 72 h) and 90% coverage, by channel")
    for c in range(4):
        s = (chan == c) & west
        cr = [P.crps_at(q, TRUE, VALID, chan, c, k, west) for k in KS]
        y, v, qq = TRUE[..., s], VALID[..., s], q[..., s, :]
        cov = 100 * (((y >= qq[..., 0]) & (y <= qq[..., 6]))[v]).mean()
        print(f"  {P.CHANNELS[c]:15s} CRPS {cr[0]:6.2f} / {cr[1]:6.2f}   coverage {cov:5.1f}%")
