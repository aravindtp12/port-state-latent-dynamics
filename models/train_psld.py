"""Train and evaluate PSLD (one run).

    python models/train_psld.py                       # train, then forecast the test origins
    python models/train_psld.py --seed 1              # a different seed
    python models/train_psld.py --intra-port-graph    # + intra-port message passing step (Eq. 9)
    python models/train_psld.py --eval-only --ckpt data/results/checkpoints/psld.pt

Recipe (Table A1): 168 h context, 72 h horizon, batch 48, AdamW (lr 2e-3, weight decay
1e-4) with a one-cycle schedule (25% warm-up), gradient clipping at 5, pinball loss
over the nine quantiles, early stopping on validation pinball (patience 10, at most
40 epochs).  Model selection uses the validation quarter only; the test quarter is
touched once, to write the forecasts.

Writes  data/results/checkpoints/<name>.pt  and  data/results/<name>_quantiles.npy
(test forecasts, [origins, 72 h, 44 series, 9 quantiles], in vessels / hours).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, json, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from psld import paths, data, protocol
from psld.model import PSLD
from psld.stgnn import build_supports

CTX, H, BATCH, PATIENCE = 168, protocol.H, 48, 10
TRAIN_STRIDE, VAL_STRIDE = 2, 6
LR, WEIGHT_DECAY, WARMUP, CLIP = 2e-3, 1e-4, 0.25, 5.0

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40, help="maximum epochs (early stopping usually ends sooner)")
ap.add_argument("--intra-port-graph", action="store_true", help="add the intra-port message-passing step")
ap.add_argument("--eval-only", action="store_true", help="skip training; forecast with --ckpt")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--name", default=None, help="output name (default: psld[_intra][_s<seed>])")
args = ap.parse_args()
NAME = args.name or ("psld" + ("_intra" if args.intra_port_graph else "") + (f"_s{args.seed}" if args.seed else ""))
CKPT_PATH = pathlib.Path(args.ckpt) if args.ckpt else paths.CKPT / f"{NAME}.pt"
paths.ensure(paths.CKPT)

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------------------------------------------ data
d = data.load()
X, U, TM = d["X"], d["U"], d["target_mask"]
T, N, _ = X.shape
tr0, tr1 = d["split"]["train"]; va0, va1 = d["split"]["val"]
dyn_i, sta_i = data.feature_split(d)
Xs = data.scale(X, d["scalers"])
Ys, med, iqr, Yanchor = data.prepare_targets(d)
Xd = torch.from_numpy(Xs[:, :, dyn_i].astype(np.float32))      # dynamic features [T,N,13]
Xst = torch.from_numpy(Xs[:, :, sta_i].astype(np.float32))     # static descriptors [T,N,8]
Ut, Yt = torch.from_numpy(U), torch.from_numpy(Ys)
Mt, At = torch.from_numpy(TM.astype(np.float32)), torch.from_numpy(Yanchor)
UU = U.shape[1]

supports = None
if args.intra_port_graph:
    supports = build_supports(data.adjacency(d, data.INTRA_PORT))
net = PSLD(len(dyn_i), len(sta_i), UU, H, n_nodes=N, supports=supports).to(dev)
n_par = sum(p.numel() for p in net.parameters())
print(f"{NAME}: {n_par:,} parameters, device {dev}")


def batch(idx):
    """Origins -> flattened (batch*nodes) tensors: history, calendar, static, anchor, target."""
    b = len(idx)
    ctx = torch.stack([Xd[t:t + CTX] for t in idx]).permute(0, 2, 1, 3).reshape(b * N, CTX, -1)
    sta = torch.stack([Xst[t + CTX - 1] for t in idx]).reshape(b * N, -1)
    anc = torch.stack([At[t + CTX - 1] for t in idx]).reshape(b * N, 4)
    uc = torch.stack([Ut[t:t + CTX] for t in idx]).unsqueeze(1).expand(b, N, CTX, UU).reshape(b * N, CTX, UU)
    uf = torch.stack([Ut[t + CTX:t + CTX + H] for t in idx]).unsqueeze(1).expand(b, N, H, UU).reshape(b * N, H, UU)
    y = torch.stack([Yt[t + CTX:t + CTX + H] for t in idx]).permute(0, 2, 1, 3).reshape(b * N, H, 4)
    m = torch.stack([Mt[t + CTX:t + CTX + H] for t in idx]).permute(0, 2, 1, 3).reshape(b * N, H, 4)
    return [z.to(dev) for z in (ctx, uc, uf, sta, anc, y, m)]


def train():
    starts = lambda a, b, s: list(range(max(a, 0), b - CTX - H + 1, s))
    tr_s = starts(tr0, tr1, TRAIN_STRIDE)
    va_s = [t for t in starts(va0 - CTX, va1, VAL_STRIDE) if t + CTX >= va0]
    print(f"  windows: train {len(tr_s):,}  val {len(va_s):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=args.epochs * math.ceil(len(tr_s) / BATCH), pct_start=WARMUP)
    best, bad, hist, t0 = math.inf, 0, [], time.time()
    for ep in range(args.epochs):
        net.train(); perm = np.random.permutation(len(tr_s)); tot = nb = 0
        for k in range(0, len(perm), BATCH):
            ctx, uc, uf, sta, anc, y, m = batch([tr_s[i] for i in perm[k:k + BATCH]])
            opt.zero_grad()
            loss = net.pinball(net(ctx, uc, uf, sta, anc), y, m)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        net.eval(); vt = vn = 0
        with torch.no_grad():
            for k in range(0, len(va_s), BATCH):
                ctx, uc, uf, sta, anc, y, m = batch(va_s[k:k + BATCH])
                vt += net.pinball(net(ctx, uc, uf, sta, anc), y, m).item(); vn += 1
        vp = vt / vn
        hist.append(dict(epoch=ep, train_pinball=tot / nb, val_pinball=vp))
        flag = ""
        if vp < best - 1e-5:
            best, bad = vp, 0; torch.save(net.state_dict(), CKPT_PATH); flag = " *"
        else:
            bad += 1
        print(f"  ep {ep:3d}  train {tot / nb:.5f}  val {vp:.5f}  [{time.time() - t0:5.0f}s]{flag}", flush=True)
        if bad >= PATIENCE:
            print("  early stop"); break
    pd.DataFrame(hist).to_csv(paths.RESULTS / f"{NAME}_history.csv", index=False)
    print(f"  best validation pinball {best:.5f} -> {CKPT_PATH}")


@torch.no_grad()
def predict(origins):
    """Quantile forecasts at the given origins -> [O, H, J, 9] in real units."""
    net.eval()
    tg = d["targets"]
    out = np.zeros((len(origins), H, len(tg), len(protocol.QUANTILES)), np.float32)
    for k in range(0, len(origins), BATCH):
        oo = [o - CTX for o in origins[k:k + BATCH]]; b = len(oo)
        ctx, uc, uf, sta, anc, _, _ = batch(oo)
        q = net(ctx, uc, uf, sta, anc).cpu().numpy().reshape(b, N, H, 4, -1).transpose(0, 2, 1, 3, 4)
        q = data.invert_targets(q, med[..., None], iqr[..., None])
        for i in range(b):
            for j, t in enumerate(tg):
                out[k + i, :, j] = q[i, :, t["node_index"], t["channel"]]
    return out


if not args.eval_only:
    train()
net.load_state_dict(torch.load(CKPT_PATH, map_location=dev))
origins = protocol.test_origins(d)
print(f"forecasting {len(origins)} test origins from the checkpoint {CKPT_PATH.name}")
protocol.save_array(f"{NAME}_quantiles.npy", predict(origins))
