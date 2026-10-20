"""Audit postproc 05: calibration of the final (corrected) H prediction.

Diagnostics: mean residual by prediction quantile (tails split finely) and by standardized distance to the lower bound.
Recalibrations, each 5-fold cross-fitted on H (fit on 4/5 of H, score on 1/5), compared with the uncalibrated chain:
  mult   y ~ c * p                                     (global multiplicative factor)
  affine y ~ c0 + c1 * p
  iso    isotonic regression of y on p
  tn2    p + k_lo * adj_lo + k_hi * adj_hi             (truncated-normal mean shift, lower / upper bound separately)
  hinge  p + sum_j beta_j * s * max(0, t_j - z_lo)     (z_lo = (p - lo) / s, t in 1.5, 2.5, 3.5, 5)  + upper tn term
All variants then get the hard clip [lo, hi] of 03.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/05_calib.py
"""
import sys

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
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
BOUNDS = {"gas": (EUIMIN * AH, np.full(len(H), GMAX)), "eui": (np.full(len(H), EUIMIN), GMAX / AH),
          "elec": (EAMIN * AH, EAMAX * AH)}


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def sd_model(m):
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    rel2 = (r / np.maximum(p, 1e-9)) ** 2
    return np.linalg.lstsq(np.column_stack([np.ones(len(p)), 1 / MP]), rel2, rcond=None)[0]


def tn_terms(p, s, lo, hi):
    a, b = (lo - p) / s, (hi - p) / s
    Z = np.clip(norm.cdf(b) - norm.cdf(a), 1e-300, None)
    return s * norm.pdf(a) / Z, -s * norm.pdf(b) / Z  # lower-bound push-up, upper-bound push-down


def cross_fit(y, p, X, lo, hi, folds):
    """p + X @ beta, beta by least squares on the fit folds; returns cross-fitted prediction (clipped)"""
    out = np.empty(len(p))
    for a, b in folds:
        beta = np.linalg.lstsq(X[a], (y - p)[a], rcond=None)[0]
        out[b] = np.clip(p[b] + X[b] @ beta, lo[b], hi[b])
    return out


folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(H))))
tot = {}
for m in ("gas", "eui", "elec"):
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    lo, hi = BOUNDS[m]
    a0, b0 = sd_model(m)
    s = p * np.sqrt(np.clip(a0 + b0 / MH, 1e-8, None))
    base = nr(y, p, m)
    hard = np.clip(p, lo, hi)
    print(f"== {m}: base {base:.6f}, hard clip {nr(y, hard, m) - base:+.7f}")
    # diagnostics by prediction quantile
    qs = [0, .001, .01, .05, .2, .5, .8, .95, .99, .999, 1]
    e = np.quantile(p, qs)
    for i in range(len(qs) - 1):
        sel = (p >= e[i]) & (p <= e[i + 1])
        r = y[sel] - p[sel]
        print(f"   p q[{qs[i]},{qs[i + 1]}] n {sel.sum():6d} mean p {p[sel].mean():12.2f} mean resid {r.mean():+11.3f} "
              f"se {r.std() / np.sqrt(sel.sum()):9.3f} t {r.mean() / (r.std() / np.sqrt(sel.sum())):+5.1f}")
    zl = (p - lo) / s
    for l, h in ((0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5), (5, 8)):
        sel = (zl >= l) & (zl < h)
        if sel.sum():
            r = y[sel] - p[sel]
            print(f"   z_lo [{l},{h}) n {sel.sum():6d} mean resid {r.mean():+11.3f} t {r.mean() / (r.std() / np.sqrt(sel.sum())):+5.1f}"
                  f" mean M {MH[sel].mean():.1f}")
    adj_lo, adj_hi = tn_terms(p, s, lo, hi)
    res = {}
    res["hard"] = hard
    res["mult"] = cross_fit(y, p, p[:, None], lo, hi, folds)
    res["affine"] = cross_fit(y, p, np.column_stack([np.ones(len(p)), p]), lo, hi, folds)
    iso = np.empty(len(p))
    for a, b in folds:
        ir = IsotonicRegression(out_of_bounds="clip").fit(p[a], y[a])
        iso[b] = np.clip(ir.predict(p[b]), lo[b], hi[b])
    res["iso"] = iso
    res["tn_hi"] = cross_fit(y, p, adj_hi[:, None], lo, hi, folds)
    res["tn2"] = cross_fit(y, p, np.column_stack([adj_lo, adj_hi]), lo, hi, folds)
    hin = np.column_stack([s * np.maximum(0, t - zl) for t in (1.5, 2.5, 3.5, 5)] + [adj_hi])
    res["hinge+tn_hi"] = cross_fit(y, p, hin, lo, hi, folds)
    # lower-tail shrink by quantile of z_lo: one coefficient on s * max(0, 4 - z_lo)
    res["hinge4+tn_hi"] = cross_fit(y, p, np.column_stack([s * np.maximum(0, 4 - zl), adj_hi]), lo, hi, folds)
    for k, v in res.items():
        d = nr(y, v, m) - base
        print(f"   {k:14s} d {d:+.7f} total {d / 5:+.8f}")
        tot[k] = tot.get(k, 0.0) + d / 5
    # coefficients of tn2 on the full H (for reference only)
    beta = np.linalg.lstsq(np.column_stack([adj_lo, adj_hi]), y - p, rcond=None)[0]
    print(f"   tn2 full-H coefficients (reference) k_lo {beta[0]:.3f} k_hi {beta[1]:.3f}")
print("TOTAL", {k: round(v, 8) for k, v in tot.items()})
