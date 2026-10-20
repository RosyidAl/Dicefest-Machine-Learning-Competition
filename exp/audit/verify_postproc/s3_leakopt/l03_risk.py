"""Adversarial check (leakage / optimism lens) of the S3 soft cap: sampling risk and shape misfit.

(a) per-distance-bin decomposition of the H gain for the claimed constants (where the rule helps / hurts)
(b) bootstrap sd of k* (H corrected; full-train OOF teacher) and the break-even point k* = k / 2
(c) realized total change of the claimed rule on random subsets of the full-train OOF (single-model regime) of the
    private-LB size (70% of test = 128,791 rows) and the public-LB size (55,196 rows)

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/s3_leakopt/l03_risk.py
"""
import json
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

t0 = time.time()
OUT = "exp/audit/verify_postproc/s3_leakopt"
GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
MET = ("gas", "eui")
KCL = {"gas": 0.5046, "eui": 0.6574}
res = {}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
GMAX_P = float((z["gas_pP"] + z["gas_rP"]).max())
SD = {}
for m in MET:
    pp, rr = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pp)), 1.0 / MP]), (rr / pp) ** 2, rcond=None)[0]


def mills_shift(p, s, T):
    b = (T - p) / s
    return -s * np.exp(norm.logpdf(b) - norm.logcdf(b))


def sdev(m, p, Mv):
    return p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / Mv, 1e-8, None))


# (a) per-bin decomposition on H (after S1, claimed constants, T from P)
pH, xH, bH, rH = {}, {}, {}, {}
for m in MET:
    p = np.clip(z[f"{m}_corrected"], 0, None)
    pp, rr = z[f"{m}_pP"], z[f"{m}_rP"]
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + float(pp[gP] @ rr[gP] / (pp[gP] @ pp[gP]))
    s = sdev(m, p, MH)
    T = GMAX_P if m == "gas" else GMAX_P / AH
    pH[m], xH[m], bH[m], rH[m] = p, mills_shift(p, s, T), (T - p) / s, z[f"{m}_y"] - p
nH = len(H)
res["a_bins"] = {}
for m in MET:
    e0 = rH[m]
    e1 = rH[m] - KCL[m] * xH[m]
    rm = np.sqrt(np.mean(e0 ** 2))
    c = (e1 ** 2 - e0 ** 2) / (2 * nH * rm * SIG[m]) / 5
    parts = []
    for lo, hi in ((-9, 0), (0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)):
        sel = (bH[m] >= lo) & (bH[m] < hi)
        parts.append((f"[{lo},{hi})", int(sel.sum()), float(c[sel].sum())))
    res["a_bins"][m] = parts
    print(f"{m} H gain by distance bin (total-score units): " + "; ".join(f"{b} n {n} {v:+.8f}" for b, n, v in parts))

# (b) bootstrap of k*
rng = np.random.default_rng(7)
kb = {m: [] for m in MET}
for _ in range(2000):
    i = rng.integers(0, nH, nH)
    for m in MET:
        kb[m].append(float(xH[m][i] @ rH[m][i] / (xH[m][i] @ xH[m][i])))
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS])
Atr, Mtr = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
GMAX = float(tr[GAS].max())
fz = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
xf, rf = {}, {}
for m in MET:
    pf = fz[f"{m}_p"]
    xf[m] = mills_shift(pf, sdev(m, pf, Mtr), GMAX if m == "gas" else GMAX / Atr)
    rf[m] = fz[f"{m}_y"] - pf
kf = {m: [] for m in MET}
n = len(Atr)
for _ in range(1000):
    i = rng.integers(0, n, n)
    for m in MET:
        kf[m].append(float(xf[m][i] @ rf[m][i] / (xf[m][i] @ xf[m][i])))
res["b_kstar"] = {}
for m in MET:
    a, f = np.array(kb[m]), np.array(kf[m])
    res["b_kstar"][m] = dict(H_mean=a.mean(), H_sd=a.std(), full_mean=f.mean(), full_sd=f.std(), breakeven=KCL[m] / 2,
                             P_H_below_breakeven=float(np.mean(a < KCL[m] / 2)), P_full_below_breakeven=float(np.mean(f < KCL[m] / 2)))
    print(f"{m}: k* H corrected {a.mean():.3f} (sd {a.std():.3f}) | full-train OOF {f.mean():.3f} (sd {f.std():.3f}) | break-even "
          f"k*={KCL[m] / 2:.3f}: P(H boot below) {np.mean(a < KCL[m] / 2):.3f}, P(full boot below) {np.mean(f < KCL[m] / 2):.3f}")

# (c) realized change of the claimed rule on random full-OOF subsets of LB size
res["c_subsets"] = {}
for name, size in (("private", 128791), ("public", 55196), ("all_train", n)):
    tots = []
    reps = 1 if name == "all_train" else 1000
    for _ in range(reps):
        i = np.arange(n) if name == "all_train" else rng.choice(n, size, replace=False)
        t = 0.0
        for m in MET:
            hi = GMAX if m == "gas" else GMAX / Atr[i]
            e0 = rf[m][i]
            p0 = fz[f"{m}_p"][i]
            p1 = np.minimum(p0 + KCL[m] * xf[m][i], hi)
            e1 = fz[f"{m}_y"][i] - p1
            t += (np.sqrt(np.mean(e1 ** 2)) - np.sqrt(np.mean(e0 ** 2))) / SIG[m] / 5
        tots.append(t)
    tots = np.array(tots)
    res["c_subsets"][name] = dict(mean=tots.mean(), sd=tots.std(), frac_neg=float(np.mean(tots < 0)),
                                  q05=float(np.quantile(tots, .05)), q95=float(np.quantile(tots, .95)))
    print(f"full-OOF {name:9s} subsets: total change mean {tots.mean():+.8f} sd {tots.std():.8f} frac<0 {np.mean(tots < 0):.3f} "
          f"5/95% [{np.quantile(tots, .05):+.8f}, {np.quantile(tots, .95):+.8f}]")
json.dump(res, open(f"{OUT}/l03_risk.json", "w"), indent=1, default=float)
print(f"wrote {OUT}/l03_risk.json; wall {time.time() - t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
