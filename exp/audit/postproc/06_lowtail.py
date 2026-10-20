"""Audit postproc 06: robustness of the lower-tail / small-M calibration effect found in 05 (EUI: hinge+tn_hi -0.000187
NRMSE). Diagnostics on H:
  - mean residual of the corrected EUI / gas prediction by M bin x prediction-quantile-within-M bin;
  - per-row share of the cross-fitted gain (is it a handful of rows?);
  - repeated random split-half (fit on one half of H, score on the other), 20 repetitions, for a few pre-declared
    feature sets.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/06_lowtail.py
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
    return s * norm.pdf(a) / Z, -s * norm.pdf(b) / Z


for m in ("eui", "gas", "elec"):
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    lo, hi = BOUNDS[m]
    a0, b0 = sd_model(m)
    s = p * np.sqrt(np.clip(a0 + b0 / MH, 1e-8, None))
    zl = (p - lo) / s
    adj_lo, adj_hi = tn_terms(p, s, lo, hi)
    r = y - p
    print(f"== {m} base {nr(y, p, m):.6f}")
    # 2D diagnostic: M bins x within-bin prediction quantile
    mb = [(2, 3), (4, 5), (6, 9), (10, 19), (20, 1000)]
    qb = [0, .01, .05, .2, .5, .8, .95, 1]
    print("   rows: M bin; cols: within-M-bin p quantile " + " ".join(f"[{qb[i]},{qb[i + 1]}]" for i in range(len(qb) - 1)))
    for l, h in mb:
        sel = (MH >= l) & (MH <= h)
        e = np.quantile(p[sel], qb)
        cells = []
        for i in range(len(qb) - 1):
            c = sel & (p >= e[i]) & (p <= e[i + 1])
            rr = r[c]
            cells.append(f"{rr.mean() / SIG[m] * 1000:+6.1f}({rr.mean() / (rr.std() / np.sqrt(c.sum())):+4.1f})")
        print(f"   M[{l},{h}] n {sel.sum():6d}: " + " ".join(cells))
    print("   (cell = mean residual in 1e-3 sigma units (t-stat))")
    feats = {
        "tn_hi": adj_hi[:, None],
        "tn2": np.column_stack([adj_lo, adj_hi]),
        "hinge+tn_hi": np.column_stack([s * np.maximum(0, t - zl) for t in (1.5, 2.5, 3.5, 5)] + [adj_hi]),
        "hinge4+tn_hi": np.column_stack([s * np.maximum(0, 4 - zl), adj_hi]),
        "invM_centered+tn_hi": np.column_stack([(p - np.median(p)) / MH, adj_hi]),
    }
    for name, X in feats.items():
        # per-row share of the 5-fold cross-fitted gain
        pc = np.empty(len(p))
        for a, b in KFold(5, shuffle=True, random_state=0).split(p):
            beta = np.linalg.lstsq(X[a], r[a], rcond=None)[0]
            pc[b] = np.clip(p[b] + X[b] @ beta, lo[b], hi[b])
        gain = (y - p) ** 2 - (y - pc) ** 2
        top = np.sort(gain)[::-1]
        d = nr(y, pc, m) - nr(y, p, m)
        # repeated split-half
        sh = []
        for rep in range(20):
            perm = np.random.default_rng(100 + rep).permutation(len(p))
            h1, h2 = perm[: len(p) // 2], perm[len(p) // 2:]
            for a, b in ((h1, h2), (h2, h1)):
                beta = np.linalg.lstsq(X[a], r[a], rcond=None)[0]
                pb = np.clip(p[b] + X[b] @ beta, lo[b], hi[b])
                sh.append(nr(y[b], pb, m) - nr(y[b], p[b], m))
        sh = np.array(sh)
        print(f"   {name:20s} 5-fold d {d:+.7f} total {d / 5:+.8f} | top1/top10/top100 rows share of gain "
              f"{top[0] / gain.sum():.2f}/{top[:10].sum() / gain.sum():.2f}/{top[:100].sum() / gain.sum():.2f} | "
              f"split-half mean {sh.mean():+.7f} sd {sh.std():.7f} frac<0 {np.mean(sh < 0):.2f}")
