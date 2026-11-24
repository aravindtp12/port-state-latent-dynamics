"""Transfer of PSLD to port complexes it never trained on (West coast).

    SCOPE=all python models/west_transfer.py zeroshot    # East+Gulf weights, unchanged
    SCOPE=all python models/west_transfer.py finetune    # then trained on West training data

PSLD shares one set of weights across all nodes and conditions each forecast on that
node's static descriptors (kind, hexagon count, area, long-run mean occupancy, approach
radius).  Here the checkpoint trained on the East+Gulf ports (models/train_psld.py) is
applied to Long Beach, Oakland and Vancouver:

  zeroshot  no weight is fitted or updated on West-coast data.  The input scalers are the
            East+Gulf ones; each West node supplies its own recent 168 h of AIS-derived
            features, its current observed state (the anchor) and its static descriptors.
            The per-node output scale (median / IQR of the log targets) and the static
            descriptors come from the West TRAINING window, which is how any new region is
            onboarded and carries no information about the test period.  This is zero-shot
            transfer of the parameters, not zero-history.
  finetune  starts from the East+Gulf weights and trains on the West nodes' training data
            at a low learning rate (5e-4, one-cycle, 20-epoch schedule).  The epoch is
            chosen on the West VALIDATION pinball, never on test.

Writes data/results/west_psld_<mode>_quantiles.npy ([origins, 72 h, series, 9 quantiles]).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, math, time
import numpy as np
import torch
import torch.nn as nn

from psld import paths, config, data, protocol
from psld.model import PSLD

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["zeroshot", "finetune"])
ap.add_argument("--base-ckpt", default=str(paths.CKPT / "psld.pt"), help="PSLD trained on East+Gulf")
ap.add_argument("--epochs", type=int, default=20, help="fine-tuning schedule length")
ap.add_argument("--lr", type=float, default=5e-4, help="fine-tuning learning rate")
args = ap.parse_args()
assert config.SCOPE == "all", "run with SCOPE=all"

CTX, H, Q, BATCH = 168, protocol.H, len(protocol.QUANTILES), 48
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)

d = data.load(); tg = d["targets"]
X, U = d["X"], d["U"]
T, N, _ = X.shape
tr0, tr1 = d["split"]["train"]; va0, va1 = d["split"]["val"]
dyn_i, sta_i = data.feature_split(d)
WEST = data.west_nodes(d)
print(f"{args.mode}: {N} nodes in the panel, {len(WEST)} West-coast nodes")

Xs = data.scale(X, d["scalers"])                               # East+Gulf input scalers, unchanged
Ys, med, iqr, Yanch = data.prepare_targets(d)                  # per-node output scale from each node's own training window
Xd = torch.from_numpy(Xs[:, :, dyn_i].astype(np.float32))
Xst = torch.from_numpy(Xs[:, :, sta_i].astype(np.float32))
Ut, At, Yt = torch.from_numpy(U), torch.from_numpy(Yanch), torch.from_numpy(Ys)
Mt = torch.from_numpy(d["target_mask"].astype(np.float32))
UU = U.shape[1]


def make():
    m = PSLD(len(dyn_i), len(sta_i), UU, H).to(dev)
    m.load_state_dict(torch.load(args.base_ckpt, map_location=dev))
    return m


def batch(idx, sub):
    b, n = len(idx), len(sub)
    ctx = torch.stack([Xd[t:t + CTX, sub] for t in idx]).permute(0, 2, 1, 3).reshape(b * n, CTX, -1)
    sta = torch.stack([Xst[t + CTX - 1, sub] for t in idx]).reshape(b * n, -1)
    anc = torch.stack([At[t + CTX - 1, sub] for t in idx]).reshape(b * n, 4)
    uc = torch.stack([Ut[t:t + CTX] for t in idx]).unsqueeze(1).expand(b, n, CTX, UU).reshape(b * n, CTX, UU)
    uf = torch.stack([Ut[t + CTX:t + CTX + H] for t in idx]).unsqueeze(1).expand(b, n, H, UU).reshape(b * n, H, UU)
    y = torch.stack([Yt[t + CTX:t + CTX + H, sub] for t in idx]).permute(0, 2, 1, 3).reshape(b * n, H, 4)
    m = torch.stack([Mt[t + CTX:t + CTX + H, sub] for t in idx]).permute(0, 2, 1, 3).reshape(b * n, H, 4)
    return [z.to(dev) for z in (ctx, uc, uf, sta, anc, y, m)]


@torch.no_grad()
def predict(net, origins):
    net.eval()
    out = np.zeros((len(origins), H, len(tg), Q), np.float32)
    for k in range(0, len(origins), BATCH):
        oo = [o - CTX for o in origins[k:k + BATCH]]; b = len(oo)
        ctx, uc, uf, sta, anc, _, _ = batch(oo, list(range(N)))
        q = net(ctx, uc, uf, sta, anc).cpu().numpy().reshape(b, N, H, 4, Q).transpose(0, 2, 1, 3, 4)
        q = data.invert_targets(q, med[..., None], iqr[..., None])
        for i in range(b):
            for j, t in enumerate(tg):
                out[k + i, :, j] = q[i, :, t["node_index"], t["channel"]]
    return out


def finetune(net, sub):
    tr_s = list(range(tr0, tr1 - CTX - H + 1, 2))
    va_s = [t for t in range(va0 - CTX, va1 - CTX - H + 1, 6) if t + CTX >= va0]
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * math.ceil(len(tr_s) / BATCH), pct_start=0.25)
    best, best_ep, t0 = float("inf"), -1, time.time()
    best_path = paths.CKPT / "west_psld_finetuned.pt"
    for ep in range(args.epochs):
        net.train(); perm = np.random.permutation(len(tr_s)); tot = nb = 0
        for k in range(0, len(perm), BATCH):
            ctx, uc, uf, sta, anc, y, m = batch([tr_s[i] for i in perm[k:k + BATCH]], sub)
            opt.zero_grad(); loss = net.pinball(net(ctx, uc, uf, sta, anc), y, m)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
        net.eval(); vt = vn = 0
        with torch.no_grad():
            for k in range(0, len(va_s), BATCH):
                ctx, uc, uf, sta, anc, y, m = batch(va_s[k:k + BATCH], sub)
                vt += net.pinball(net(ctx, uc, uf, sta, anc), y, m).item(); vn += 1
        flag = ""
        if vt / vn < best:
            best, best_ep = vt / vn, ep
            torch.save(net.state_dict(), best_path); flag = " *"
        print(f"  ep {ep:3d}  train {tot / nb:.5f}  val {vt / vn:.5f}  [{time.time() - t0:5.0f}s]{flag}", flush=True)
    print(f"  best West validation pinball {best:.5f} at epoch {best_ep + 1}")
    net.load_state_dict(torch.load(best_path, map_location=dev))
    return net


net = make()
if args.mode == "finetune":
    net = finetune(net, WEST)
origins = protocol.test_origins(d)
protocol.save_array(f"west_psld_{'zeroshot' if args.mode == 'zeroshot' else 'finetuned'}_quantiles.npy",
                    predict(net, origins))
