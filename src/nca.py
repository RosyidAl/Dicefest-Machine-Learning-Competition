"""NCA0001: retrieval regressor in the style of ModernNCA (Ye et al. 2024). An MLP embeds every row; the prediction of
a row is the softmax(-distance)-weighted mean of the (standardised) targets of the training rows, so the network learns
which rows are neighbours instead of using the hand-made similarity spaces of the kNN-target features. Training:
mini-batches of query rows against a random share of the training rows as candidates (own row masked); prediction:
all training rows of the fold as candidates. Same folds, inputs (base + aggregates + spatial / kNN encodings) and
target formulations as the MLP members (src/nn.py run_nn); fixed epoch count, no look at the validation rows.
"""
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
from common import TARGET_COLS, nrmse  # noqa: E402
from nn import PLR, preprocess, set_seed, te_block  # noqa: E402
from train_lib import is_gas_kind, is_log_kind, target_spec  # noqa: E402


class Embed(nn.Module):
    def __init__(self, d_in, hidden=(512,), d_out=128, drop=0.1, emb=None, cont=None):
        super().__init__()
        if emb is not None:
            self.register_buffer("ci", torch.tensor(np.flatnonzero(cont)))
            self.register_buffer("bi", torch.tensor(np.flatnonzero(~cont)))
            self.emb = PLR(len(self.ci), **emb)
            d_in = len(self.ci) * emb.get("d", 24) + len(self.bi)
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self, "emb"):
            x = torch.cat([self.emb(x[:, self.ci]), x[:, self.bi]], 1)
        return self.net(x)


def _dist(q, c):
    """euclidean distances (explicit formula: deterministic matmul instead of cdist)"""
    d2 = (q * q).sum(1, keepdim=True) + (c * c).sum(1)[None] - 2.0 * q @ c.T
    return torch.sqrt(d2.clamp_min(1e-8))


def fit_fold_nca(Za, ya, wa, Zb, Zt, epochs=20, bs=512, lr=2e-3, wd=1e-5, hidden=(512,), d_out=128, drop=0.1,
                 emb=None, cand=0.1, dev="cuda", order_seed=42, chunk=128):
    mu, sd = ya.mean(), ya.std()
    g = torch.Generator(device="cpu").manual_seed(order_seed)
    cont = None if emb is None else np.array([len(np.unique(Za[:, j])) > 2 for j in range(Za.shape[1])])
    model = Embed(Za.shape[1], hidden, d_out, drop, emb, cont).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(Za) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xa = torch.tensor(Za, device=dev)
    Ya = torch.tensor((ya - mu) / sd, dtype=torch.float32, device=dev)
    Wa = torch.tensor(np.ones(len(ya)) if wa is None else wa / np.mean(wa), dtype=torch.float32, device=dev)
    n, nc = len(Za), int(cand * len(Za))
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            q = perm[i:i + bs]
            c = torch.randperm(n, generator=g)[:nc].to(dev)
            e = model(torch.cat([Xa[q], Xa[c]]))
            logit = -_dist(e[:len(q)], e[len(q):])
            logit = logit.masked_fill(q[:, None] == c[None], -1e9)  # a row is never its own neighbour
            pred = (torch.softmax(logit, 1) * Ya[c][None]).sum(1)
            lv = (Wa[q] * (pred - Ya[q]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            lv.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        Ea = torch.cat([model(Xa[i:i + 65536]) for i in range(0, n, 65536)])

        def predict(Z):
            X = torch.tensor(Z, device=dev)
            out = []
            for i in range(0, len(X), chunk):  # chunk x n_train distance matrix per step (GPU is shared)
                logit = -_dist(model(X[i:i + chunk]), Ea)
                out.append((torch.softmax(logit, 1) * Ya[None]).sum(1))
            return torch.cat(out).cpu().numpy() * sd + mu

        return predict(Zb), predict(Zt), epochs - 1


def run_nca(d, name, kind, out_dir="exp", wmul=None, use_te=True, windows=(2, 10), cells=True, knn=True,
            extra=None, extra_t=None, seeds=(42,), folds=None, **kw):
    set_seed()
    if extra is None:
        Z, Zt = preprocess(d.X, d.Xt)
    else:
        Z, Zt = preprocess(pd.concat([d.X.reset_index(drop=True), extra], axis=1),
                           pd.concat([d.Xt.reset_index(drop=True), extra_t], axis=1))
    y, w, mult = target_spec(d, kind)
    mult_t = target_spec(d, kind, test=True)[2]
    if wmul is not None:
        w = wmul / wmul.mean() if w is None else w * wmul / np.mean(w * wmul)
    oof, pt = np.zeros(len(Z)), np.zeros(len(Zt))
    t0 = time.time()
    fl = d.folds if folds is None else [d.folds[k] for k in folds]
    for k, (a, b) in enumerate(fl):
        Za, Zb, Ztk = Z[a], Z[b], Zt
        if use_te:
            Ta, Tb, Tt = te_block(d, a, b, windows, cells, knn)
            Za, Zb, Ztk = np.hstack([Za, Ta]), np.hstack([Zb, Tb]), np.hstack([Zt, Tt])
        for s in seeds:
            set_seed(s + k)
            pb, ptk, _ = fit_fold_nca(Za, y[a], None if w is None else w[a], Zb, Ztk, order_seed=1000 * s + k, **kw)
            if is_log_kind(kind):
                pb, ptk = np.exp(pb), np.exp(ptk)
            oof[b] += pb * (mult if np.isscalar(mult) else mult[b]) / len(seeds)
            pt += ptk * mult_t / len(fl) / len(seeds)
        print(f"  [{name}] fold {k} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{out_dir}/oof", exist_ok=True)
    os.makedirs(f"{out_dir}/pred", exist_ok=True)
    np.save(f"{out_dir}/oof/{name}.npy", oof)
    np.save(f"{out_dir}/pred/{name}.npy", pt)
    done = np.zeros(len(oof), bool)
    for _, b in fl:
        done[b] = True
    if is_gas_kind(kind):
        msg = (f"eui={nrmse(d.y[TARGET_COLS[2]][done], oof[done], TARGET_COLS[2]):.5f} "
               f"gas={nrmse(d.y[TARGET_COLS[0]][done], oof[done] * d.A[done], TARGET_COLS[0]):.5f}")
    else:
        msg = f"elec={nrmse(d.y[TARGET_COLS[1]][done], oof[done], TARGET_COLS[1]):.5f}"
    print(f"[{name}] {msg} rows={done.sum()} time={time.time() - t0:.0f}s", flush=True)
    return oof, pt
