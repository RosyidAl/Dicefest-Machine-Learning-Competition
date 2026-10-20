"""verify2_segments: the gas cap calibration on FULL-train OOF predictions (models trained on 90% of all train rows,
closer to the test-time models than the P-only holdout chain).

Base predictions: the 27 v19vce gas members (exp/oof, EUI scale), blended two ways with my own low-memory NNLS:
  global NNLS (gas: row_scale A; eui: unscaled) and the production varying-coefficient spec A2xM2xp2 (Gram-matrix NNLS).
Blend weights are in-sample on all rows (slightly optimistic base, irrelevant for the k honesty); k is always fitted on
one part of the rows and scored on another: P->H of common.holdout_split, 5-fold x5 seeds, 2-fold x50 seeds and
leave-one-region-out over 10 contiguous sub_n blocks. Effects are TOTAL-score changes (per-target NRMSE change / 5).
"""
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.special import log_ndtr

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402  (rank-based hat basis only; the NNLS below is my own)
from common import holdout_split  # noqa: E402

GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
CAP = 1e6
names = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12", "subzone_id"])
yg, ye = tr[GAS].values.astype(float), tr[EUI].values.astype(float)
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
subn = tr["subzone_id"].str[4:].astype(int).values
del tr
Pf = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
n = len(yg)
print(f"{len(names)} gas members, {n} rows")


def gram_nnls(cols, y, s, chunk=40000):
    """NNLS of y*s on X*s, X built chunk-wise by cols(slice); solved via the Cholesky factor of the column-normalised
    Gram matrix (same minimiser as the direct NNLS up to rounding)."""
    G, c = None, None
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(n, i0 + chunk))
        X = cols(sl) * s[sl, None]
        G = X.T @ X if G is None else G + X.T @ X
        c = X.T @ (y[sl] * s[sl]) if c is None else c + X.T @ (y[sl] * s[sl])
    d = np.sqrt(np.diag(G))
    Gn = G / np.outer(d, d) + 1e-12 * np.eye(len(d))
    L = np.linalg.cholesky(Gn)
    b = solve_triangular(L, c / d, lower=True)
    w, _ = nnls(L.T, b, maxiter=50 * len(d))
    return w / d


def blends(metric):
    s = A if metric == "gas" else np.ones(n)
    w = nnls(Pf * s[:, None], ye * s)[0]
    wg = gram_nnls(lambda sl: Pf[sl], ye, s)
    p_glob = Pf @ w
    print(f"  {metric}: global NNLS direct vs Gram max |dw| {np.abs(w - wg).max():.2e}")
    B = vc_basis("A2xM2xp2", dict(A=A, M=M, p=p_glob), dict(A=A, M=M, p=p_glob))
    nb, nm = B.shape[1], Pf.shape[1]
    W = gram_nnls(lambda sl: np.concatenate([Pf[sl] * B[sl, [j]] for j in range(nb)], axis=1), ye, s)
    W = W.reshape(nb, nm).T
    p_vc = ((Pf @ W) * B).sum(axis=1)
    if metric == "gas":
        return p_glob * A, p_vc * A
    return p_glob, p_vc


def mills(a):
    return np.exp(-0.5 * a * a - 0.5 * np.log(2.0 * np.pi) - log_ndtr(a))


def calib(g, k):
    out = np.array(g, dtype=float, copy=True)
    if k <= 0:
        return out
    m = out > 5e5
    s = k * out[m]
    out[m] = out[m] - s * mills((CAP - out[m]) / s)
    return out


def calib_eui(e, area, k):
    g = e * area
    gc = calib(g, k)
    out = np.array(e, dtype=float, copy=True)
    m = gc != g
    out[m] = e[m] * gc[m] / g[m]
    return out


KGRID = np.round(np.arange(0.0, 0.3001, 0.0025), 4)


def fit_k(p, y, area=None):
    m = (p * (area if area is not None else 1.0)) > 5e5
    best, bk = None, 0.0
    for k in KGRID:
        q = calib_eui(p[m], area[m], k) if area is not None else calib(p[m], k)
        sse = np.sum((y[m] - q) ** 2)
        if best is None or sse < best - 1e-9:
            best, bk = sse, k
    return float(bk)


def nrmse(y, p, sig):
    return float(np.sqrt(np.mean((y - p) ** 2)) / sig)


def crossfit(p, y, folds, sig, area=None):
    q = p.copy()
    ks = []
    for f in folds:
        trn = np.ones(len(p), bool)
        trn[f] = False
        k = fit_k(p[trn], y[trn], area[trn] if area is not None else None)
        q[f] = calib_eui(p[f], area[f], k) if area is not None else calib(p[f], k)
        ks.append(k)
    return (nrmse(y, q, sig) - nrmse(y, p, sig)) / 5, ks


P, H = holdout_split(n)
order = np.argsort(subn, kind="stable")
regions = np.array_split(order, 10)
edges = [-np.inf, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, np.inf]
for metric in ("gas", "eui"):
    pg, pv = blends(metric)
    y = yg if metric == "gas" else ye
    area = None if metric == "gas" else A
    for label, p in (("global NNLS", pg), ("VC A2xM2xp2", pv)):
        print(f"\n== {metric} {label}: NRMSE {nrmse(y, p, SIG[metric]):.6f}")
        lvl = p * (1.0 if area is None else area) / 1e6
        r = y * (1.0 if area is None else area) - p * (1.0 if area is None else area)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (lvl >= lo) & (lvl < hi)
            if mm.sum() > 1:
                print(f"   level [{lo:5.2f},{hi:5.2f}) n {mm.sum():6d}  mean resid (gas units) {r[mm].mean():+10.0f}"
                      f"  t {r[mm].mean() / (r[mm].std() / np.sqrt(mm.sum())):+6.2f}")
        kP = fit_k(p[P], y[P], None if area is None else A[P])
        qH = calib(p[H], kP) if area is None else calib_eui(p[H], A[H], kP)
        dPH = (nrmse(y[H], qH, SIG[metric]) - nrmse(y[H], p[H], SIG[metric])) / 5
        kH = fit_k(p[H], y[H], None if area is None else A[H])
        qP = calib(p[P], kH) if area is None else calib_eui(p[P], A[P], kH)
        dHP = (nrmse(y[P], qP, SIG[metric]) - nrmse(y[P], p[P], SIG[metric])) / 5
        print(f"   P->H: k {kP} dTotal on H {dPH:+.8f} | H->P: k {kH} dTotal on P {dHP:+.8f}")
        for kf in (0.08, 0.085):
            q = calib(p, kf) if area is None else calib_eui(p, A, kf)
            print(f"   fixed k {kf}: all-row dTotal {(nrmse(y, q, SIG[metric]) - nrmse(y, p, SIG[metric])) / 5:+.8f}")
        r5 = [crossfit(p, y, np.array_split(np.random.default_rng(700 + s).permutation(n), 5), SIG[metric], area)[0]
              for s in range(5)]
        r2 = [crossfit(p, y, np.array_split(np.random.default_rng(900 + s).permutation(n), 2), SIG[metric], area)[0]
              for s in range(50)]
        dreg, kreg = crossfit(p, y, regions, SIG[metric], area)
        print(f"   5-fold x5 cross-fit: mean {np.mean(r5):+.8f} sd {np.std(r5):.8f} | 2-fold x50: mean {np.mean(r2):+.8f}"
              f" sd {np.std(r2):.8f} worst {np.max(r2):+.8f} | leave-region-out {dreg:+.8f} (k {min(kreg)}..{max(kreg)})")
