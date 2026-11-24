"""Graph WaveNet and DCRNN, written from scratch for this dataset.

Both follow their papers (Wu et al. 2019; Li et al. 2018) and the reference
implementations in Bigscity-LibCity, with changes forced by this problem rather
than by preference:

  * 48 input steps instead of 12, so Graph WaveNet needs a receptive field of
    at least 48 -- 2 blocks of 5 layers (dilations 1,2,4,8,16) give 63.
  * four output channels per node, not one, so the final 1x1 emits
    horizon x channels and is reshaped.
  * the adjacency is the measured port graph (intra-port structure + O-D
    voyages) rather than a road-network distance kernel.

Each model comes in two output variants: a point head (trained on masked MAE) and
a nine-quantile head (trained on pinball loss) that is identical to PSLD's, so the
CRPS comparison isolates the architecture from having a distribution at all.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def asym_norm(A):
    """Row-normalised random-walk matrix, the transition operator both papers use."""
    d = A.sum(1); d[d == 0] = 1.0
    return (A / d[:, None]).astype(np.float32)


def build_supports(A):
    """Forward and backward transition matrices with self-loops."""
    A = np.asarray(A, dtype=np.float64) + np.eye(len(A))
    return [asym_norm(A), asym_norm(A.T)]


# --------------------------------------------------------------- Graph WaveNet
class NConv(nn.Module):
    def forward(self, x, adj):                       # x: [B,C,N,T]
        return torch.einsum("bcvt,vw->bcwt", x, adj).contiguous()


class GCN(nn.Module):
    """Diffusion step of order `order` over every support, then a 1x1 mix."""
    def __init__(self, c_in, c_out, dropout, support_len, order=2):
        super().__init__()
        self.nconv = NConv()
        self.mlp = nn.Conv2d((order*support_len + 1)*c_in, c_out, (1, 1))
        self.dropout, self.order = dropout, order

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a); out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a); out.append(x1)
        return F.dropout(self.mlp(torch.cat(out, 1)), self.dropout, self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, in_dim, horizon, out_dim, supports,
                 blocks=2, layers=5, kernel=2, residual=32, dilation=32,
                 skip=256, end=512, dropout=0.3, adapt_dim=10, n_q=1):
        super().__init__()
        self.n_q = n_q
        self.horizon, self.out_dim, self.dropout = horizon, out_dim, dropout
        self.blocks, self.layers, self.kernel = blocks, layers, kernel
        self.register_buffer("supports", torch.stack(
            [torch.tensor(s) for s in supports]) if supports else torch.zeros(0))
        # predefined supports (forward / backward transitions) + the learned adaptive one
        n_sup = len(supports) + 1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, adapt_dim)*0.1)
        self.nodevec2 = nn.Parameter(torch.randn(adapt_dim, num_nodes)*0.1)
        self.start = nn.Conv2d(in_dim, residual, (1, 1))
        self.filt, self.gate, self.res, self.skipc, self.bn, self.gconv = (
            nn.ModuleList() for _ in range(6))
        rf = 1
        for _ in range(blocks):
            d = 1
            for _ in range(layers):
                self.filt.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.gate.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.res.append(nn.Conv2d(dilation, residual, (1, 1)))
                self.skipc.append(nn.Conv2d(dilation, skip, (1, 1)))
                self.bn.append(nn.BatchNorm2d(residual))
                self.gconv.append(GCN(dilation, residual, dropout, n_sup))
                rf += d*(kernel-1); d *= 2
        self.n_sup = n_sup
        self.receptive_field = rf
        self.end1 = nn.Conv2d(skip, end, (1, 1))
        self.end2 = nn.Conv2d(end, horizon*out_dim*n_q, (1, 1))

    def _supports(self):
        sup = list(self.supports) if len(self.supports) else []
        return sup + [F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)]

    def forward(self, x, y_last=None):         # x: [B,T,N,F]; y_last unused (kept for a common call signature)
        B, T, N, _ = x.shape
        x = x.permute(0, 3, 2, 1)              # [B, F, N, T]
        if T < self.receptive_field:
            x = F.pad(x, (self.receptive_field - T, 0, 0, 0))
        x = self.start(x)
        sup = self._supports()
        skip = 0
        for i in range(self.blocks*self.layers):
            res = x
            s = torch.tanh(self.filt[i](res)) * torch.sigmoid(self.gate[i](res))
            sk = self.skipc[i](s)
            skip = sk if isinstance(skip, int) else skip[..., -sk.size(3):] + sk
            x = self.gconv[i](s, sup)
            x = x + res[..., -x.size(3):]
            x = self.bn[i](x)
        # the original head: relu -> 1x1 -> relu -> 1x1
        h = F.relu(self.end1(F.relu(skip[..., -1:])))
        h = self.end2(h)                        # [B, horizon*out_dim, N, 1]
        if self.n_q > 1:                         # quantile head: [B,H,N,C,Q]
            o = h.squeeze(-1).reshape(B, self.horizon, self.out_dim*self.n_q, N).permute(0, 1, 3, 2)
            return monotone_quantiles(o, self.out_dim, self.n_q)
        return h.squeeze(-1).reshape(B, self.horizon, self.out_dim, N).permute(0, 1, 3, 2)


# ----------------------------------------------------------------------- DCRNN
class DiffusionConv(nn.Module):
    """Sum_k theta_k T_k(A) x, T_k the k-step random walk, per Li et al. 2018."""
    def __init__(self, c_in, c_out, supports, K=2, bias_start=0.0):
        super().__init__()
        self.K = K
        self.register_buffer("sup", torch.stack([torch.tensor(s) for s in supports]))
        n_mat = len(supports)*K + 1
        self.w = nn.Parameter(torch.empty(c_in*n_mat, c_out))
        nn.init.xavier_normal_(self.w)
        self.b = nn.Parameter(torch.full((c_out,), bias_start))

    def forward(self, x):                      # x: [B, N, c_in]
        out = [x]
        for a in self.sup:
            x1 = torch.einsum("vw,bwc->bvc", a, x); out.append(x1)
            for _ in range(2, self.K+1):
                x1 = torch.einsum("vw,bwc->bvc", a, x1); out.append(x1)
        return torch.einsum("bvc,co->bvo", torch.cat(out, -1), self.w) + self.b


class DCGRUCell(nn.Module):
    def __init__(self, in_dim, hid, supports, K=2):
        super().__init__()
        self.hid = hid
        self.gates = DiffusionConv(in_dim+hid, 2*hid, supports, K, bias_start=1.0)
        self.cand = DiffusionConv(in_dim+hid, hid, supports, K)

    def forward(self, x, h):                   # x: [B,N,in], h: [B,N,hid]
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], -1))).chunk(2, -1)
        c = torch.tanh(self.cand(torch.cat([x, r*h], -1)))
        return u*h + (1-u)*c


def monotone_quantiles(o, C, Q):
    """[..., C*Q] raw outputs -> [..., C, Q] non-decreasing quantiles: a free base
    level plus strictly positive softplus increments, the same construction PSLD
    uses, so no head can emit a crossing."""
    o = o.reshape(*o.shape[:-1], C, Q)
    base, inc = o[..., :1], F.softplus(o[..., 1:])
    return torch.cat([base, base + torch.cumsum(inc, -1)], -1)


class DCRNN(nn.Module):
    """Encoder-decoder over the graph, with scheduled sampling during training.

    n_q > 1 attaches a quantile head: proj emits out_dim*n_q raw values per step
    and the decoder feeds its own MEDIAN (index `mid`) back as the next input, so
    the recurrence still sees an out_dim-wide signal.  n_q == 1 is the original
    point model, bit for bit."""
    def __init__(self, num_nodes, in_dim, horizon, out_dim, supports,
                 hid=64, n_layers=2, K=2, cl_decay=2000, n_q=1, mid=0):
        super().__init__()
        self.n_q, self.mid = n_q, mid
        self.N, self.horizon, self.out_dim = num_nodes, horizon, out_dim
        self.hid, self.n_layers, self.cl_decay = hid, n_layers, cl_decay
        self.enc = nn.ModuleList([DCGRUCell(in_dim if i == 0 else hid, hid, supports, K)
                                  for i in range(n_layers)])
        self.dec = nn.ModuleList([DCGRUCell(out_dim if i == 0 else hid, hid, supports, K)
                                  for i in range(n_layers)])
        self.proj = nn.Linear(hid, out_dim*n_q)

    def forward(self, x, y=None, step=0):      # x: [B,T,N,F]  y: [B,H,N,C]
        B = x.size(0)
        h = [torch.zeros(B, self.N, self.hid, device=x.device) for _ in range(self.n_layers)]
        for t in range(x.size(1)):
            inp = x[:, t]
            for l, cell in enumerate(self.enc):
                h[l] = cell(inp, h[l]); inp = h[l]
        out, go = [], torch.zeros(B, self.N, self.out_dim, device=x.device)
        for t in range(self.horizon):
            inp = go
            for l, cell in enumerate(self.dec):
                h[l] = cell(inp, h[l]); inp = h[l]
            go = self.proj(inp); out.append(go)
            if self.n_q > 1:                     # feed back the median, not all quantiles
                go = monotone_quantiles(go, self.out_dim, self.n_q)[..., self.mid]
            if self.training and y is not None:          # scheduled sampling
                eps = self.cl_decay/(self.cl_decay + np.exp(step/self.cl_decay))
                if np.random.rand() < eps:
                    go = y[:, t]
        out = torch.stack(out, 1)                          # [B,H,N,out_dim*n_q]
        return monotone_quantiles(out, self.out_dim, self.n_q) if self.n_q > 1 else out
