"""PSLD: Port State Latent-Dynamics world model.

A shared GRU encoder compresses a node's last week of hourly observations into a
latent state, which is rolled forward one hour at a time (72 steps) under calendar
forcing and decoded into nine monotone quantiles per channel and hour.

    encoder     GRU over [x_t'; U_t'] for the L = 168 history hours -> h
    start       z0 = MLP([h; static descriptors; anchor y_t])
    rollout     z_k = LayerNorm(GRUCell([U_{t+k}; static; anchor; phi(k)], z_{k-1}))
    (optional)  z_k <- z_k + graph-mixing step over the intra-port edges (Eq. 9)
    decoder     MLP([z_k; static; anchor; phi(k)]) -> 4 channels x 9 quantiles,
                monotone by construction: a free base level plus softplus increments

One set of weights is shared by every node; port scale enters only through the
static descriptors, which is what lets the model be applied to unseen ports.
The reported model is the graph-free one (no `supports`).  With 13 dynamic features,
7 calendar features and 8 static descriptors it has 166,404 parameters.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from psld.protocol import QUANTILES


class GraphMix(nn.Module):
    """One diffusion-convolution message-passing step over the port graph.

    Order-2 diffusion over each support, concatenated and mixed by a 1x1, then added
    as a RESIDUAL with the mixing layer zero-initialised: at step 0 the model is
    exactly the graph-free model, so every later difference is attributable to
    message passing."""

    def __init__(self, latent, n_nodes, supports, order=2):
        super().__init__()
        self.N, self.order = n_nodes, order
        self.register_buffer("sup", torch.stack([torch.tensor(s) for s in supports]))
        self.mix = nn.Linear(latent * (order * len(supports) + 1), latent)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, z):                       # z: [B*N, latent]
        B = z.size(0) // self.N
        h = z.view(B, self.N, -1)
        out = [h]
        for a in self.sup:
            x = h
            for _ in range(self.order):
                x = torch.einsum("vw,bwc->bvc", a, x)
                out.append(x)
        return z + self.mix(torch.cat(out, -1)).reshape(B * self.N, -1)


class PSLD(nn.Module):
    def __init__(self, n_dyn, n_static, n_u, horizon, n_chan=4, quantiles=QUANTILES,
                 hid=128, latent=96, dropout=0.1, n_nodes=None, supports=None):
        super().__init__()
        self.H, self.C, self.Q = horizon, n_chan, len(quantiles)
        self.register_buffer("qs", torch.tensor(quantiles, dtype=torch.float32))
        ke = torch.arange(horizon, dtype=torch.float32) / horizon
        # horizon encoding phi(k) = [(k-1)/H, sin(pi (k-1)/H), cos(pi (k-1)/H)]
        self.register_buffer("kenc", torch.stack([ke, torch.sin(np.pi * ke), torch.cos(np.pi * ke)], -1))
        aux = n_chan + 3                        # anchor + horizon encoding
        self.enc = nn.GRU(n_dyn + n_u, hid, 1, batch_first=True)
        self.to_latent = nn.Sequential(nn.Linear(hid + n_static + n_chan, 2 * latent), nn.SiLU(),
                                       nn.Dropout(dropout), nn.Linear(2 * latent, latent))
        self.cell = nn.GRUCell(n_u + n_static + aux, latent)
        self.norm = nn.LayerNorm(latent)
        self.head = nn.Sequential(nn.Linear(latent + n_static + aux, 2 * latent), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(2 * latent, n_chan * self.Q))
        self.gm = GraphMix(latent, n_nodes, supports) if supports else None

    def forward(self, x_dyn, u_ctx, u_fut, stat, anchor):
        """x_dyn [B,L,D]  u_ctx [B,L,U]  u_fut [B,H,U]  stat [B,S]  anchor [B,C]
        -> quantiles [B,H,C,Q] in standardised log space."""
        B = x_dyn.size(0)
        _, h = self.enc(torch.cat([x_dyn, u_ctx], -1))
        z = self.to_latent(torch.cat([h[-1], stat, anchor], -1))
        outs = []
        for k in range(self.H):
            aux = [anchor, self.kenc[k].expand(B, 3)]
            z = self.norm(self.cell(torch.cat([u_fut[:, k], stat] + aux, -1), z))
            if self.gm is not None:
                z = self.gm(z)
            outs.append(self.head(torch.cat([z, stat] + aux, -1)))
        o = torch.stack(outs, 1).view(B, self.H, self.C, self.Q)
        base, inc = o[..., :1], F.softplus(o[..., 1:])
        return torch.cat([base, base + torch.cumsum(inc, -1)], -1)

    def pinball(self, q_pred, y, mask):
        """q_pred [B,H,C,Q]  y [B,H,C]  mask [B,H,C] -> masked mean pinball loss."""
        e = y.unsqueeze(-1) - q_pred
        loss = torch.maximum(self.qs * e, (self.qs - 1.0) * e).mean(-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1)
