"""Independent re-verification (round 'iv'): the truncation bias and the calibration gain on the full-train OOF of the
27 v19vce gas-family members, blended by my own global NNLS (gas row-scaled by A, eui direct). k is cross-fitted with
GroupKFold by zone (spatially disjoint, harder than the claim's random 5-fold) and by subzone.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_segments/iv3_fulloof.py
"""
import json
import resource
import time

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar, nnls
from scipy.special import erfcx

t0 = time.time()
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
CAP = 1e6
names = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
assert len(names) == 27
tr = pd.read_parquet("cache/train.parquet", columns=["zone_region_id", "subzone_id", "bld_dim_06", GAS, EUI])
A = tr["bld_dim_06"].values
yg, ye = tr[GAS].values, tr[EUI].values
zone = tr["zone_region_id"].values
sub = tr["subzone_id"].values
X = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])  # eui units
wg, _ = nnls(X * A[:, None], yg)
we, _ = nnls(X, ye)
pg = X @ wg * A
pe = X @ we
del X
print(f"members {len(names)}; NNLS gas weight sum {wg.sum():.4f} nonzero {(wg > 0).sum()}; eui sum {we.sum():.4f} nonzero {(we > 0).sum()}")
print(f"full-OOF global NNLS: gas NRMSE {np.sqrt(np.mean((yg - pg) ** 2)) / SIG['gas']:.6f}  eui {np.sqrt(np.mean((ye - pe) ** 2)) / SIG['eui']:.6f}")


def calib(pred, k, scale):
    g = pred * scale
    res = pred.copy()
    m = g > 6e5
    if m.any():
        gm = g[m]
        s = k * gm
        a = (CAP - gm) / s
        res[m] = pred[m] * (gm - s * np.sqrt(2.0 / np.pi) / erfcx(-a / np.sqrt(2.0))) / gm
    return res


def fit_k(y, pred, scale):
    m = pred * scale > 6e5
    yy, pp, ss = y[m], pred[m], scale[m]

    def f(k):
        return float(np.sum((yy - calib(pp, k, ss)) ** 2))

    grid = np.arange(0.005, 0.3001, 0.005)
    i = int(np.argmin([f(k) for k in grid]))
    return float(minimize_scalar(f, bounds=(max(grid[i] - 0.005, 1e-3), grid[i] + 0.005), method="bounded",
                                 options={"xatol": 1e-6}).x)


def dtot(y, base, new, met):
    n = len(y)
    return (np.sqrt(np.sum((y - new) ** 2) / n) - np.sqrt(np.sum((y - base) ** 2) / n)) / SIG[met] / 5


res = {}
for met, y, p, sc in (("gas", yg, pg, np.ones(len(A))), ("eui", ye, pe, A)):
    g = p * sc
    r = (y - p) * sc
    print(f"\n== {met}: bias (gas units) by level/1e6")
    for lo, hi in ((0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0), (1.0, np.inf)):
        m = (g >= lo * 1e6) & (g < hi * 1e6)
        print(f"   [{lo:.2f},{hi:.2f}) n {m.sum():4d} mean {r[m].mean():+9.0f} t {r[m].mean() / (r[m].std(ddof=1) / np.sqrt(m.sum())):+5.2f}")
    o = res[met] = {}
    for gname, gv in (("zone", zone), ("subzone", sub)):
        u, inv = np.unique(gv, return_inverse=True)
        reps, ks = [], []
        for seed in range(7101, 7111):
            lab = np.random.default_rng(seed).integers(0, 5, len(u))[inv]
            new = p.copy()
            for j in range(5):
                te_ = lab == j
                k = fit_k(y[~te_], p[~te_], sc[~te_])
                ks.append(k)
                new[te_] = calib(p[te_], k, sc[te_])
            reps.append(dtot(y, p, new, met))
        reps = np.array(reps)
        o[gname] = dict(mean=float(reps.mean()), sd=float(reps.std(ddof=1)), worst=float(reps.max()),
                        kmin=float(min(ks)), kmax=float(max(ks)))
        print(f"   GroupKFold({gname}) 5x10 cross-fit: dTotal mean {reps.mean():+.8f} sd {reps.std(ddof=1):.8f} "
              f"worst {reps.max():+.8f}  k {min(ks):.4f}..{max(ks):.4f}  ({len(u)} groups)")
    kall = fit_k(y, p, sc)
    o["k_all"] = kall
    print(f"   k fitted on all rows {kall:.4f}; fixed k 0.08 in-sample dTotal {dtot(y, p, calib(p, 0.08, sc), met):+.8f}")
json.dump(res, open("exp/audit/verify2_segments/iv3_fulloof.json", "w"), indent=1)
print(f"\nelapsed {time.time() - t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
