"""Variants of the gas-cap truncation calibration (robustness), honest scores.

V1  imr k            : s = k p                              (baseline, one parameter)
V2  imr k1,k2 (M)    : s = p sqrt(k1^2 + k2 / M)           (household-averaging noise, two parameters)
V3  imr k, cap 999765: cap = train max instead of 1e6
EUI: separate k fitted on the EUI teacher (g = eui_pred * A against the gas truth) vs the gas k.
Scores: (a) fit on P-side teacher, score on H corrected; (b) 5-fold x 10 cross-fit on H corrected.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_cap2.py > exp/audit/segments/seg_cap2.log
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
gas = pd.read_parquet("cache/train.parquet", columns=["target_macro_gas_kwh"])["target_macro_gas_kwh"].values


def adj(p, s, cap):
    s = np.maximum(s, 1e-9)
    a = (cap - p) / s
    return p - s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def variants():
    yield "V1 k", [dict(k=k) for k in np.arange(0.04, 0.161, 0.005)], lambda p, m, q: adj(p, q["k"] * p, 1e6)
    yield ("V2 k1,k2/M", [dict(k1=a, k2=b) for a, b in itertools.product(np.arange(0.0, 0.141, 0.01), [0.0, 0.1, 0.2, 0.4, 0.8, 1.6])],
           lambda p, m, q: adj(p, p * np.sqrt(q["k1"] ** 2 + q["k2"] / m), 1e6))
    yield "V3 k cap=max", [dict(k=k) for k in np.arange(0.04, 0.161, 0.005)], lambda p, m, q: adj(p, q["k"] * p, 999765.0)


def fit(f, grid, y, p, m):
    sse = [np.sum((y - f(p, m, q)) ** 2) for q in grid]
    return grid[int(np.argmin(sse))]


def nr(e, k):
    return np.sqrt(np.mean(e ** 2)) / SIG[k]


yP, pP, MP = gas[P], z["gas_pP"], M[P]
yH, pc, MH, AH = z["gas_y"], z["gas_corrected"], M[H], A[H]
# restrict fits/evaluations to rows where any variant can matter (p > 0.5e6); the rest are unchanged by construction
selP, selH = pP > 5e5, pc > 5e5
base = nr(yH - pc, "gas")
for name, grid, f in variants():
    q = fit(f, grid, yP[selP], pP[selP], MP[selP])
    pn = pc.copy()
    pn[selH] = f(pc[selH], MH[selH], q)
    d_p = (nr(yH - pn, "gas") - base) / 5
    outs = []
    for rep in range(10):
        pn2 = pc.copy()
        for a, b in KFold(5, shuffle=True, random_state=rep).split(yH):
            a2, b2 = a[selH[a]], b[selH[b]]
            qq = fit(f, grid, yH[a2], pc[a2], MH[a2])
            pn2[b2] = f(pc[b2], MH[b2], qq)
        outs.append((nr(yH - pn2, "gas") - base) / 5)
    print(f"gas {name:14s} P-fit {q} -> H dTotal {d_p:+.8f} | H cross-fit {np.mean(outs):+.8f} (sd {np.std(outs):.8f})", flush=True)

# EUI: same-k (gas teacher fit) vs own k fitted on the EUI teacher (g = eui * A against gas truth)
yE, pE = z["eui_y"], z["eui_corrected"]
gEP = z["eui_pP"] * A[P]
gEH = pE * AH
baseE = nr(yE - pE, "eui")
grid = [dict(k=k) for k in np.arange(0.04, 0.161, 0.005)]
f1 = lambda p, m, q: adj(p, q["k"] * p, 1e6)  # noqa: E731
qE = fit(f1, grid, yP[gEP > 5e5], gEP[gEP > 5e5], MP[gEP > 5e5])
for label, q in [("gas k 0.085", dict(k=0.085)), (f"own P-fit {qE}", qE)]:
    sel = gEH > 5e5
    gn = gEH.copy()
    gn[sel] = f1(gEH[sel], None, q)
    print(f"eui via eui*A, {label}: H dTotal {(nr(yE - pE * gn / gEH, 'eui') - baseE) / 5:+.8f}")
# EUI-weighted fit: minimise the EUI-scale error (eui = g / A) instead of the gas-scale error
sse = []
sel = gEP > 5e5
for q in grid:
    gn = f1(gEP[sel], None, q)
    sse.append(np.sum(((yP[sel] - gn) / A[P][sel]) ** 2))
qW = grid[int(np.argmin(sse))]
sel = gEH > 5e5
gn = gEH.copy()
gn[sel] = f1(gEH[sel], None, qW)
print(f"eui via eui*A, EUI-scale P-fit {qW}: H dTotal {(nr(yE - pE * gn / gEH, 'eui') - baseE) / 5:+.8f}")
