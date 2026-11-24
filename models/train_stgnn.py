"""Train and evaluate the graph neural network baselines, Graph WaveNet and DCRNN (one run).

    python models/train_stgnn.py gwnet                          # point forecasts (Table I)
    python models/train_stgnn.py dcrnn
    python models/train_stgnn.py gwnet --head quantile          # nine-quantile head (Table II)
    python models/train_stgnn.py dcrnn --head quantile
    SCOPE=all python models/train_stgnn.py gwnet --domain west  # in-domain West-coast fit (Table III)

Both networks are written from scratch in psld/stgnn.py following their papers; they see
the same data, splits, validation-based early stopping and test origins as PSLD, with
two differences that follow the original designs: a 48 h input window and no known-ahead
calendar (the decoders read the phase off the history).

  point     loss = masked MAE on standardised log targets (Table A1 configuration)
  quantile  the same networks with PSLD's nine-quantile output head (base + cumulative
            softplus increments, so no quantile crossing) and pinball loss; model
            selection on validation pinball.  DCRNN feeds its own median back as the next
            decoder input.

Common settings: Adam (lr 1e-3, weight decay 1e-4), ReduceLROnPlateau (x0.5, patience 4),
gradient clipping at 5, batch 64, early stopping patience 10 (max 60 epochs).

DCRNN uses scheduled sampling with decay constant `--cl-decay`.  The original paper's
2000 is the default for the point head.  With 2000 the teacher-forcing probability stays
near 1 for the whole run and a quantile head's free-running errors compound at test time,
so the quantile head defaults to 20.

--domain west trains on the three West-coast complexes' nodes only (the East+Gulf graph
cannot be applied there).  No measured West graph exists, so the predefined supports are
the identity and Graph WaveNet runs on its learned adaptive adjacency alone.  This needs
the all-scope tensors (SCOPE=all) and is a point-head experiment.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from psld import paths, config, data, protocol, stgnn

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["gwnet", "dcrnn"])
ap.add_argument("--head", choices=["point", "quantile"], default="point")
ap.add_argument("--domain", choices=["east", "west"], default="east")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--cl-decay", type=float, default=None, help="DCRNN scheduled-sampling decay constant")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--name", default=None)
args = ap.parse_args()
assert not (args.domain == "west" and args.head == "quantile"), "West-coast baselines use the point head"
assert (args.domain == "west") == (config.SCOPE == "all"), "--domain west needs SCOPE=all (and east needs the default scope)"
QUANT = args.head == "quantile"
CL = args.cl_decay if args.cl_decay is not None else (20.0 if QUANT else 2000.0)
NAME = args.name or ("west_" if args.domain == "west" else "") + f"{args.model}_{args.head}" + (f"_s{args.seed}" if args.seed else "")
CKPT_PATH = pathlib.Path(args.ckpt) if args.ckpt else paths.CKPT / f"{NAME}.pt"
paths.ensure(paths.CKPT)

L, H, BATCH, PATIENCE, TRAIN_STRIDE, VAL_STRIDE = 48, protocol.H, 64, 10, 2, 6
MIN_GAIN = 1e-5 if QUANT else 1e-4          # validation improvement that resets early stopping
QS, Q = protocol.QUANTILES, len(protocol.QUANTILES)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------------------------------------------ data
d = data.load()
X, U, TM = d["X"], d["U"], d["target_mask"]
order, tgts = d["nodes"], d["targets"]
T, N, _ = X.shape
tr0, tr1 = d["split"]["train"]; va0, va1 = d["split"]["val"]
SUB = data.west_nodes(d) if args.domain == "west" else list(range(N))      # nodes the network is fit on
pos = {g: k for k, g in enumerate(SUB)}; n = len(SUB)
print(f"{NAME}: {n} nodes, device {dev}, epochs <= {args.epochs}")

Xs = data.scale(X, d["scalers"])
Xs = np.concatenate([Xs, np.repeat(U[:, None, :], N, axis=1)], axis=2).astype(np.float32)   # + calendar
Ys, med, iqr, Yanchor = data.prepare_targets(d)
Xt, Yt = torch.from_numpy(Xs[:, SUB]), torch.from_numpy(Ys[:, SUB])
Mt, At = torch.from_numpy(TM[:, SUB].astype(np.float32)), torch.from_numpy(Yanchor[:, SUB])
F_IN = Xt.shape[2]


def batch(idx):
    return (torch.stack([Xt[t:t + L] for t in idx]).to(dev),
            torch.stack([Yt[t + L:t + L + H] for t in idx]).to(dev),
            torch.stack([Mt[t + L:t + L + H] for t in idx]).to(dev),
            torch.stack([At[t + L - 1] for t in idx]).to(dev))


# ------------------------------------------------------------------ network
if args.domain == "west":
    supports = stgnn.build_supports(np.zeros((n, n)))                      # identity: no West graph exists
else:
    supports = stgnn.build_supports(data.adjacency(d))                     # measured intra- and inter-port graph
if args.model == "gwnet":
    net = stgnn.GraphWaveNet(n, F_IN, H, 4, supports, n_q=Q if QUANT else 1)
else:
    net = stgnn.DCRNN(n, F_IN, H, 4, supports, cl_decay=CL, **(dict(n_q=Q, mid=protocol.MEDIAN) if QUANT else {}))
net = net.to(dev)
n_par = sum(p.numel() for p in net.parameters())
print(f"  {n_par:,} parameters" + (f", scheduled-sampling decay {CL:g}" if args.model == "dcrnn" else ""))

qs_t = torch.tensor(QS, dtype=torch.float32, device=dev)


def loss_fn(p, y, m):
    if QUANT:                                     # p [B,H,N,C,Q]; pinball averaged over the nine levels
        e = y.unsqueeze(-1) - p
        l = torch.maximum(qs_t * e, (qs_t - 1.0) * e).mean(-1)
    else:                                         # p [B,H,N,C]; absolute error
        l = torch.abs(p - y)
    return (l * m).sum() / m.sum().clamp(min=1)


def train():
    starts = lambda a, b, s: list(range(a, b - L - H + 1, s))
    tr_s, va_s = starts(tr0, tr1, TRAIN_STRIDE), starts(va0, va1, VAL_STRIDE)
    print(f"  windows: train {len(tr_s):,}  val {len(va_s):,}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    best, bad, step, t0, hist = math.inf, 0, 0, time.time(), []
    for ep in range(args.epochs):
        net.train(); perm = np.random.permutation(len(tr_s)); tot = nb = 0
        for k in range(0, len(perm), BATCH):
            ctx, tgt, msk, anc = batch([tr_s[i] for i in perm[k:k + BATCH]])
            opt.zero_grad()
            out = net(ctx, tgt, step) if args.model == "dcrnn" else net(ctx, anc)
            loss = loss_fn(out, tgt, msk); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1; step += 1
        net.eval(); vt = vn = 0
        with torch.no_grad():
            for k in range(0, len(va_s), BATCH):
                ctx, tgt, msk, anc = batch(va_s[k:k + BATCH])
                vt += loss_fn(net(ctx, anc), tgt, msk).item(); vn += 1
        tr_l, va_l = tot / nb, vt / vn
        sched.step(va_l); hist.append(dict(epoch=ep, train=tr_l, val=va_l)); flag = ""
        if va_l < best - MIN_GAIN:
            best, bad = va_l, 0; torch.save(net.state_dict(), CKPT_PATH); flag = " *"
        else:
            bad += 1
        print(f"  ep {ep:3d}  train {tr_l:.4f}  val {va_l:.4f}  [{time.time() - t0:5.0f}s]{flag}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop (no validation gain in {PATIENCE} epochs)"); break
    pd.DataFrame(hist).to_csv(paths.RESULTS / f"{NAME}_history.csv", index=False)
    print(f"  best validation loss {best:.4f} -> {CKPT_PATH}")


@torch.no_grad()
def predict(origins):
    """-> [O,H,J] point forecasts, or [O,H,J,9] quantiles, in real units (NaN for series
    of nodes the network was not fit on)."""
    net.eval()
    shape = (len(origins), H, len(tgts)) + ((Q,) if QUANT else ())
    P = np.full(shape, np.nan, np.float32)
    for k in range(0, len(origins), BATCH):
        oo = origins[k:k + BATCH]
        ctx = torch.stack([Xt[o - L:o] for o in oo]).to(dev)
        anc = torch.stack([At[o - 1] for o in oo]).to(dev)
        out = net(ctx, anc).cpu().numpy()                                   # [b,H,n,4(,Q)]
        sc = (lambda a: a[SUB, :, None]) if QUANT else (lambda a: a[SUB])
        out = data.invert_targets(out, sc(med), sc(iqr))
        for b in range(len(oo)):
            for j, t in enumerate(tgts):
                if t["node_index"] in pos:
                    P[k + b, :, j] = out[b, :, pos[t["node_index"]], t["channel"]]
    return P


if not args.eval_only:
    train()
net.load_state_dict(torch.load(CKPT_PATH, map_location=dev))
origins = protocol.test_origins(d)
print(f"forecasting {len(origins)} test origins from the checkpoint {CKPT_PATH.name}")
P = predict(origins)
protocol.save_array(f"{NAME}_quantiles.npy" if QUANT else f"{NAME}_predictions.npy", P)
