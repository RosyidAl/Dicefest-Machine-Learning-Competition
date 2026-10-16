"""Deterministic MLP regressor on the same folds as the GBMs (for ensemble diversity)."""
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, TARGET_COLS, nrmse  # noqa: E402
from train_lib import is_gas_kind, is_log_kind, target_spec  # noqa: E402


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def preprocess(X, Xt):
    """log1p on non-negative skewed columns, standardise with train statistics, NaN -> 0 plus flags."""
    X = X.copy()
    Xt = Xt.copy()
    for c in X.columns:
        v = X[c].values
        if np.nanmin(v) >= 0 and np.nanmax(v) > 0 and (np.nanmean(v) > 0) and \
                (np.nanpercentile(v, 99) > 5 * max(np.nanmedian(v), 1e-6)):
            X[c] = np.log1p(X[c])
            Xt[c] = np.log1p(Xt[c])
    nan_cols = [c for c in X.columns if X[c].isna().any() or Xt[c].isna().any()]
    mu = X.mean()
    sd = X.std().replace(0, 1)
    Z = ((X - mu) / sd).clip(-6, 6)
    Zt = ((Xt - mu) / sd).clip(-6, 6)
    Z = pd.concat([Z, X[nan_cols].isna().astype("float32").add_suffix("_nan")], axis=1)
    Zt = pd.concat([Zt, Xt[nan_cols].isna().astype("float32").add_suffix("_nan")], axis=1)
    return Z.fillna(0).values.astype("float32"), Zt.fillna(0).values.astype("float32")


class PLR(nn.Module):
    """Periodic embeddings of numerical features (Gorishniy et al. 2022): per feature
    x -> ReLU(Linear([sin(2 pi c x), cos(2 pi c x)])) with trainable frequencies c."""

    def __init__(self, n_feat, k=24, sigma=0.05, d=24):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_feat, k) * sigma)
        bound = 1 / np.sqrt(2 * k)
        self.w = nn.Parameter(torch.empty(n_feat, 2 * k, d).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(n_feat, d).uniform_(-bound, bound))

    def forward(self, x):
        v = 2 * np.pi * self.c[None] * x[..., None]
        z = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return torch.relu(torch.einsum("bfk,fkd->bfd", z, self.w) + self.b[None]).flatten(1)


class MLP(nn.Module):
    """emb=None: plain MLP on the inputs; emb=dict(k, sigma, d): PLR embeddings of the non-binary columns
    (cont = boolean mask over input columns), binary columns passed through.
    ens=k > 0: TabM-mini (Gorishniy et al. 2024) - k members share the backbone; each member sees the input
    times its own random-sign vector and has its own output head. forward then returns (batch, k).
    n_out > 1 (plain head only): multi-task output, forward returns (batch, n_out)."""

    def __init__(self, d_in, hidden=(512, 256, 128), drop=0.1, emb=None, cont=None, ens=0, n_out=1):
        super().__init__()
        if emb is not None:
            self.register_buffer("ci", torch.tensor(np.flatnonzero(cont)))
            self.register_buffer("bi", torch.tensor(np.flatnonzero(~cont)))
            self.emb = PLR(len(self.ci), **emb)
            d_in = len(self.ci) * emb.get("d", 24) + len(self.bi)
        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]
            d = h
        self.ens = ens
        if ens:
            self.r = nn.Parameter(torch.randint(0, 2, (ens, d_in)).float() * 2 - 1)
            bound = 1 / np.sqrt(d)
            self.head_w = nn.Parameter(torch.empty(ens, d).uniform_(-bound, bound))
            self.head_b = nn.Parameter(torch.empty(ens).uniform_(-bound, bound))
        else:
            layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self, "emb"):
            x = torch.cat([self.emb(x[:, self.ci]), x[:, self.bi]], 1)
        if self.ens:
            h = self.net((x[:, None, :] * self.r[None]).flatten(0, 1)).view(len(x), self.ens, -1)
            return torch.einsum("bkd,kd->bk", h, self.head_w) + self.head_b
        return self.net(x).squeeze(-1)


def predict_model(model, Xs, emb, ens, mu, sd):
    """eval-mode prediction in fixed chunks (chunk sizes are part of the numerics), back on the target scale"""
    model.eval()
    with torch.no_grad():
        ch = 65536 if emb is None else 8192  # PLR activations are ~50x wider
        if ens:
            ch = 1024
        out = torch.cat([model(Xs[i:i + ch]) for i in range(0, len(Xs), ch)])
        if ens:
            out = out.mean(1)
    model.train()
    return out.cpu().numpy() * sd + mu


def fit_fold(Za, ya, wa, Zb, yb, wb, Zt, epochs=30, bs=4096, lr=2e-3, wd=1e-5, hidden=(512, 256, 128),
             drop=0.1, dev="cuda", fixed_ep=None, emb=None, order_seed=SEED, ens=0, save_path=None):
    """fixed_ep=None keeps the epoch with the best validation score (optimistic OOF);
    fixed_ep=k stops after epoch k of the same schedule without looking at the validation rows.
    emb: PLR embedding settings (see MLP), None for the plain MLP.
    order_seed: seed of the mini-batch order (the default SEED gives every seed and fold the same order).
    ens: TabM-mini members (see MLP); the loss averages the members' losses, predictions average the members.
    ya / wa of shape (n, t): multi-task fit of t targets with per-task weights (needs fixed_ep)."""
    multi = ya.ndim == 2
    mu, sd = (ya.mean(0), ya.std(0)) if multi else (ya.mean(), ya.std())
    g = torch.Generator(device="cpu").manual_seed(order_seed)
    cont = None if emb is None else np.array([len(np.unique(Za[:, j])) > 2 for j in range(Za.shape[1])])
    model = MLP(Za.shape[1], hidden, drop, emb, cont, ens, n_out=ya.shape[1] if multi else 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(Za) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xa = torch.tensor(Za, device=dev)
    Ya = torch.tensor((ya - mu) / sd, dtype=torch.float32, device=dev)
    Wa = torch.tensor(np.ones(ya.shape) if wa is None else wa, dtype=torch.float32, device=dev)
    Xb = torch.tensor(Zb, device=dev)
    Xt = torch.tensor(Zt, device=dev)
    wbn = np.ones(len(yb)) if wb is None else wb

    def predict(Xs):
        return predict_model(model, Xs, emb, ens, mu, sd)

    best, best_state, best_ep = np.inf, None, -1
    for ep in range(epochs):
        perm = torch.randperm(len(Za), generator=g).to(dev)
        for i in range(0, len(Za), bs):
            idx = perm[i:i + bs]
            out = model(Xa[idx])
            if ens:
                loss = (Wa[idx, None] * (out - Ya[idx, None]) ** 2).mean()
            else:
                loss = (Wa[idx] * (out - Ya[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if fixed_ep is not None:
            if ep == fixed_ep:
                if save_path:
                    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "cont": cont,
                                "arch": dict(d_in=Za.shape[1], hidden=tuple(hidden), drop=drop, emb=emb, ens=ens,
                                             n_out=ya.shape[1] if multi else 1)}, save_path + ".pt")
                return predict(Xb), predict(Xt), ep
            continue
        pb = predict(Xb)
        score = np.sqrt(np.average((pb - yb) ** 2, weights=wbn))
        if score < best:
            best, best_ep = score, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(Xb), predict(Xt), best_ep


def te_block(d, a, b, windows, cells=False):
    """Per-fold spatial encodings, standardised with fitting-part statistics."""
    Fa = d.te_features(a, "oof", windows, cells)
    Fb = d.te_features(a, d.te_apply(b), windows, cells)
    Ft = d.te_features(a, d.te_apply(), windows, cells)
    cnt = [c for c in Fa.columns if c.endswith("_cnt")]
    for F in (Fa, Fb, Ft):
        F[cnt] = np.log1p(F[cnt])
    mu, sd = Fa.mean(), Fa.std().replace(0, 1)
    out = []
    for F in (Fa, Fb, Ft):
        Z = ((F - mu) / sd).clip(-6, 6)
        nanf = F.isna().astype("float32").add_suffix("_nan")
        out.append(np.hstack([Z.fillna(0).values, nanf.values]).astype("float32"))
    return out


def run_nn(d, name, kind, out_dir="exp", wmul=None, use_te=False, windows=(), seeds=(SEED,), cells=False,
           vary_order=False, extra=None, extra_t=None, model_dir=None, **kw):
    """vary_order: mini-batch order seeded per (seed, fold) instead of one shared order.
    extra / extra_t: additional feature frames for train / test rows (e.g. Data.agg_features())."""
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
    oof = np.zeros(len(Z))
    pt = np.zeros(len(Zt))
    t0 = time.time()
    eps = []
    for k, (a, b) in enumerate(d.folds):
        Za, Zb, Ztk = Z[a], Z[b], Zt
        if use_te:
            Ta, Tb, Tt = te_block(d, a, b, windows, cells)
            Za, Zb, Ztk = np.hstack([Za, Ta]), np.hstack([Zb, Tb]), np.hstack([Zt, Tt])
        for s in seeds:
            set_seed(s + k)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            pb, ptk, ep = fit_fold(Za, y[a], None if w is None else w[a], Zb, y[b],
                                   None if w is None else w[b], Ztk,
                                   save_path=f"{model_dir}/fold{k}_seed{s}" if model_dir else None,
                                   **(dict(kw, order_seed=1000 * s + k) if vary_order else kw))
            if is_log_kind(kind):
                pb, ptk = np.exp(pb), np.exp(ptk)
            oof[b] += pb * (mult if np.isscalar(mult) else mult[b]) / len(seeds)
            pt += ptk * mult_t / len(d.folds) / len(seeds)
            eps.append(ep)
        print(f"  [{name}] fold {k} best_ep={eps[-len(seeds):]} {time.time() - t0:.0f}s", flush=True)
    np.save(f"{out_dir}/oof/{name}.npy", oof)
    np.save(f"{out_dir}/pred/{name}.npy", pt)
    if is_gas_kind(kind):
        msg = (f"eui={nrmse(d.y[TARGET_COLS[2]], oof, TARGET_COLS[2]):.5f} "
               f"gas={nrmse(d.y[TARGET_COLS[0]], oof * d.A, TARGET_COLS[0]):.5f}")
    else:
        msg = f"elec={nrmse(d.y[TARGET_COLS[1]], oof, TARGET_COLS[1]):.5f}"
    print(f"[{name}] {msg} epochs={eps} time={time.time() - t0:.0f}s", flush=True)
    return oof, pt


def run_nn_multi(d, name, tasks, out_dir="exp", use_te=False, windows=(), seeds=(SEED,), cells=False, extra=None,
                 extra_t=None, model_dir=None, **kw):
    """Multi-task MLP: one network fits several target formulations at once (shared trunk, one output each).
    tasks: list of (suffix, kind, wmul); each task's weights are normalised to mean 1, so the tasks count equally.
    Saves one OOF / test prediction per task as <name>_<suffix> (same physical scales as run_nn)."""
    set_seed()
    if extra is None:
        Z, Zt = preprocess(d.X, d.Xt)
    else:
        Z, Zt = preprocess(pd.concat([d.X.reset_index(drop=True), extra], axis=1),
                           pd.concat([d.Xt.reset_index(drop=True), extra_t], axis=1))
    Ys, Ws, mults, mults_t = [], [], [], []
    for _, kind, wmul in tasks:
        y, w, mult = target_spec(d, kind)
        if wmul is not None:
            w = wmul if w is None else w * wmul
        Ys.append(y)
        Ws.append(np.ones(len(y)) if w is None else w / np.mean(w))
        mults.append(mult)
        mults_t.append(target_spec(d, kind, test=True)[2])
    Y, W = np.column_stack(Ys), np.column_stack(Ws)
    oof = np.zeros((len(Z), len(tasks)))
    pt = np.zeros((len(Zt), len(tasks)))
    t0 = time.time()
    for k, (a, b) in enumerate(d.folds):
        Za, Zb, Ztk = Z[a], Z[b], Zt
        if use_te:
            Ta, Tb, Tt = te_block(d, a, b, windows, cells)
            Za, Zb, Ztk = np.hstack([Za, Ta]), np.hstack([Zb, Tb]), np.hstack([Zt, Tt])
        for s in seeds:
            set_seed(s + k)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            pb, ptk, _ = fit_fold(Za, Y[a], W[a], Zb, Y[b], W[b], Ztk,
                                  save_path=f"{model_dir}/fold{k}_seed{s}" if model_dir else None, **kw)
            for j, (_, kind, _) in enumerate(tasks):
                qb, qt = (np.exp(pb[:, j]), np.exp(ptk[:, j])) if is_log_kind(kind) else (pb[:, j], ptk[:, j])
                m = mults[j]
                oof[b, j] += qb * (m if np.isscalar(m) else m[b]) / len(seeds)
                pt[:, j] += qt * mults_t[j] / len(d.folds) / len(seeds)
        print(f"  [{name}] fold {k} {time.time() - t0:.0f}s", flush=True)
    msg = []
    for j, (suffix, kind, _) in enumerate(tasks):
        np.save(f"{out_dir}/oof/{name}_{suffix}.npy", oof[:, j])
        np.save(f"{out_dir}/pred/{name}_{suffix}.npy", pt[:, j])
        if is_gas_kind(kind):
            msg.append(f"{suffix}: eui={nrmse(d.y[TARGET_COLS[2]], oof[:, j], TARGET_COLS[2]):.5f} "
                       f"gas={nrmse(d.y[TARGET_COLS[0]], oof[:, j] * d.A, TARGET_COLS[0]):.5f}")
        else:
            msg.append(f"{suffix}: elec={nrmse(d.y[TARGET_COLS[1]], oof[:, j], TARGET_COLS[1]):.5f}")
    print(f"[{name}] {' | '.join(msg)} time={time.time() - t0:.0f}s", flush=True)
    return oof, pt

