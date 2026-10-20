"""Audit postproc 04: soft truncation adjustment near the hard target bounds (gas < 1e6, elec/A < 150, elec/A >= 5,
EUI >= 5) on the honest H chain predictions.

1. Mean residual of the corrected prediction on H binned by distance to the bound (in units of the predictive sd).
2. Truncated-normal conditional mean: p' = p - k * s * phi(a) / Phi(a), a = (T - p) / s (upper bound T);
   s = p * sqrt(a0 + b0 / M) with (a0, b0) fitted on P-OOF residuals (log-residual variance vs 1/M);
   k fitted cross-fitted (5-fold) on H by least squares, also k = 1 fixed (no H fit at all).
   Then the hard clip of 03 on top.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/04_softcap.py
"""
import sys

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], Mall[P], Mall[H]
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in SIG}
GMAX = yP["gas"].max()
EAMAX, EAMIN, EUIMIN = (yP["elec"] / AP).max(), (yP["elec"] / AP).min(), yP["eui"].min()


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def var_model(m):
    """log-residual variance a0 + b0/M fitted on P-OOF (relative residual r/p squared vs 1/M)"""
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    rel2 = (r / np.maximum(p, 1e-9)) ** 2
    X = np.column_stack([np.ones(len(p)), 1 / MP])
    coef = np.linalg.lstsq(X, rel2, rcond=None)[0]
    return coef


def trunc_adj(p, s, lo=None, hi=None):
    """E[Y | lo <= Y < hi] - p for Y ~ N(p, s^2)"""
    a = (lo - p) / s if lo is not None else np.full(len(p), -np.inf)
    bnd = (hi - p) / s if hi is not None else np.full(len(p), np.inf)
    Z = np.clip(norm.cdf(bnd) - norm.cdf(a), 1e-300, None)
    return s * (norm.pdf(a) - norm.pdf(bnd)) / Z


setups = {
    "gas": (lambda p: (EUIMIN * AH, GMAX)),
    "eui": (lambda p: (EUIMIN, GMAX / AH)),
    "elec": (lambda p: (EAMIN * AH, EAMAX * AH)),
}
total_fixed, total_cf = 0.0, 0.0
for m in ("gas", "eui", "elec"):
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    a0, b0 = var_model(m)
    s = p * np.sqrt(np.clip(a0 + b0 / MH, 1e-8, None))
    lo, hi = setups[m](p)
    s0 = nr(y, p, m)
    # 1. residual bias by standardized distance to the upper bound
    dist = (hi - p) / s
    print(f"== {m}: a0 {a0:.5f} b0 {b0:.4f}; base {s0:.6f}")
    for l, h in ((-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)):
        sel = (dist >= l) & (dist < h)
        if sel.sum():
            r = y[sel] - p[sel]
            print(f"   upper dist [{l},{h}) n {sel.sum():6d} mean resid {r.mean():12.3f} (se {r.std() / np.sqrt(sel.sum()):10.3f})"
                  f"  mean resid/s {np.mean(r / s[sel]):+.3f}  pred adj/s {np.mean(trunc_adj(p[sel], s[sel], lo if np.ndim(lo) == 0 else lo[sel], hi if np.ndim(hi) == 0 else hi[sel]) / s[sel]):+.3f}")
    distl = (p - lo) / s
    for l, h in ((-9, 0), (0, 1), (1, 2), (2, 3)):
        sel = (distl >= l) & (distl < h)
        if sel.sum():
            r = y[sel] - p[sel]
            print(f"   lower dist [{l},{h}) n {sel.sum():6d} mean resid {r.mean():12.3f} mean resid/s {np.mean(r / s[sel]):+.3f}")
    # 2. truncated-normal adjustment
    adj = trunc_adj(p, s, lo, hi)
    for k in (0.25, 0.5, 1.0):
        pk = np.clip(p + k * adj, lo, hi)
        print(f"   trunc-normal k={k}: {nr(y, pk, m):.6f}  d {nr(y, pk, m) - s0:+.7f} total {(nr(y, pk, m) - s0) / 5:+.8f}")
    pk1 = np.clip(p + adj, lo, hi)
    total_fixed += (nr(y, pk1, m) - s0) / 5
    pc = np.empty(len(p))
    ks = []
    for f, (tr_, te_) in enumerate(KFold(5, shuffle=True, random_state=0).split(p)):
        x, t = adj[tr_], (y - p)[tr_]
        k = float(x @ t / max(x @ x, 1e-30))
        ks.append(round(k, 3))
        pc[te_] = np.clip(p[te_] + k * adj[te_], lo if np.ndim(lo) == 0 else lo[te_], hi if np.ndim(hi) == 0 else hi[te_])
    d = nr(y, pc, m) - s0
    total_cf += d / 5
    hard = np.clip(p, lo, hi)
    print(f"   cross-fitted k {ks}: d {d:+.7f} total {d / 5:+.8f}; hard clip only d {nr(y, hard, m) - s0:+.7f}")
print(f"TOTAL trunc-normal k=1 (no H fit) {total_fixed:+.8f}; cross-fitted k {total_cf:+.8f}")
