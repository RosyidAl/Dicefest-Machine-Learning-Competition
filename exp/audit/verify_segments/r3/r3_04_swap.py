"""verify_segments r3: value of REPLACING the adopted D015 soft cap S3 with the claimed IMR calibration (H chain).

Both rules fitted honestly (P teacher -> H corrected, and 5-fold x 20 cross-fit inside H). Paired bootstrap over H rows
of the difference (IMR - S3), at H size and at public size; and an importance-weighted (test prediction mix) value.
Effects are TOTAL-score changes (per-target NRMSE change / 5).
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/r3/r3_04_swap.py \
         > exp/audit/verify_segments/r3/r3_04_swap.log 2>&1
"""
import resource
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr
from scipy.stats import norm

T0 = time.time()
SIG = {"gas": 151750.04, "eui": 40.65}
CAP, THR = 1e6, 6e5
L2PI = 0.5 * np.log(2 * np.pi)


def cal(g, k):
    g = np.asarray(g, dtype=float)
    out = g.copy()
    m = g > THR
    s = k * g[m]
    a = (CAP - g[m]) / s
    out[m] = g[m] - s * np.exp(-0.5 * a * a - L2PI - log_ndtr(a))
    return out


def cal_eui(e, Ae, k):
    g = e * Ae
    gn = cal(g, k)
    return np.where(g > THR, e * gn / np.where(g > THR, g, 1.0), e)


z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
MP, MH, AH, AP = M[P], M[H], A[H], A[P]
GMAX_P = float((z["gas_pP"] + z["gas_rP"]).max())
SD = {}
for m in ("gas", "eui"):
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(p)), 1 / MP]), (r / p) ** 2, rcond=None)[0]


def tn_x(p, Mv, Av, m):
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / Mv, 1e-8, None))
    hi = GMAX_P if m == "gas" else GMAX_P / Av
    bb = (hi - p) / s
    return -s * norm.pdf(bb) / np.clip(norm.cdf(bb), 1e-300, None)


def fit_imr(y, p, kind, Av=None):
    if kind == "gas":
        mm = p > THR
        f = lambda k: float(np.sum((y[mm] - cal(p[mm], k)) ** 2))  # noqa: E731
    else:
        mm = p * Av > THR
        f = lambda k: float(np.sum((y[mm] - cal_eui(p[mm], Av[mm], k)) ** 2))  # noqa: E731
    grid = np.arange(0.005, 0.3001, 0.0025)
    k0 = grid[int(np.argmin([f(k) for k in grid]))]
    return float(minimize_scalar(f, bounds=(max(0.001, k0 - 0.003), k0 + 0.003), method="bounded").x)


yH = {m: z[f"{m}_y"] for m in ("gas", "eui")}
pH = {m: np.clip(z[f"{m}_corrected"], 0, None) for m in ("gas", "eui")}
xH = {m: tn_x(pH[m], MH, AH, m) for m in ("gas", "eui")}
# P-fit versions (claim: one gas-scale k for both columns)
kI = fit_imr(z["gas_pP"] + z["gas_rP"], np.clip(z["gas_pP"], 0, None), "gas")
pred = {"IMR": {}, "S3": {}}
for m in ("gas", "eui"):
    pP = np.clip(z[f"{m}_pP"], 0, None)
    yP = z[f"{m}_pP"] + z[f"{m}_rP"]
    xP = tn_x(pP, MP, AP, m)
    k3 = float(xP @ (yP - pP) / (xP @ xP))
    pred["S3"][m] = pH[m] + k3 * xH[m]
    pred["IMR"][m] = cal(pH[m], kI) if m == "gas" else cal_eui(pH[m], AH, kI)
se = {r: {m: (yH[m] - pred[r][m]) ** 2 for m in ("gas", "eui")} for r in pred}
se0 = {m: (yH[m] - pH[m]) ** 2 for m in ("gas", "eui")}


def tot(idx, rule):
    out = 0.0
    for m in ("gas", "eui"):
        a = se[rule][m] if idx is None else se[rule][m][idx]
        b = se0[m] if idx is None else se0[m][idx]
        out += (np.sqrt(a.mean()) - np.sqrt(b.mean())) / SIG[m] / 5
    return out


print(f"P-fit: IMR total {tot(None, 'IMR'):+.8f}  S3 total {tot(None, 'S3'):+.8f}  "
      f"difference IMR - S3 {tot(None, 'IMR') - tot(None, 'S3'):+.8f}")
rng = np.random.default_rng(11)
for nm, sz in (("H", len(H)), ("public", 55196)):
    dif = []
    for _ in range(3000):
        i = rng.integers(0, len(H), sz)
        dif.append(tot(i, "IMR") - tot(i, "S3"))
    dif = np.array(dif)
    print(f"  paired bootstrap {nm:6s} size {sz}: IMR - S3 mean {dif.mean():+.8f} sd {dif.std():.8f} "
          f"P(IMR better) {np.mean(dif < 0):.3f}")

# cross-fitted inside H, both rules, 5-fold x 20
difs = []
for rep in range(20):
    perm = np.random.default_rng(500 + rep).permutation(len(H))
    cfp = {"IMR": {m: pH[m].copy() for m in yH}, "S3": {m: pH[m].copy() for m in yH}}
    for part in np.array_split(perm, 5):
        trn = np.setdiff1d(perm, part)
        kg = fit_imr(yH["gas"][trn], pH["gas"][trn], "gas")
        cfp["IMR"]["gas"][part] = cal(pH["gas"][part], kg)
        cfp["IMR"]["eui"][part] = cal_eui(pH["eui"][part], AH[part], kg)
        for m in yH:
            k3 = float(xH[m][trn] @ (yH[m][trn] - pH[m][trn]) / (xH[m][trn] @ xH[m][trn]))
            cfp["S3"][m][part] = pH[m][part] + k3 * xH[m][part]
    v = {}
    for rl in cfp:
        v[rl] = sum((np.sqrt(np.mean((yH[m] - cfp[rl][m]) ** 2)) - np.sqrt(np.mean(se0[m]))) / SIG[m] / 5 for m in yH)
    difs.append((v["IMR"], v["S3"]))
difs = np.array(difs)
print(f"5-fold x 20 cross-fit inside H: IMR {difs[:, 0].mean():+.8f}  S3 {difs[:, 1].mean():+.8f}  "
      f"IMR - S3 mean {np.mean(difs[:, 0] - difs[:, 1]):+.8f} (sd over reps {np.std(difs[:, 0] - difs[:, 1]):.8f})")

# importance-weighted to the test prediction mix (sub_v19vce gas level)
g_t = pd.read_csv("submissions/sub_v19vce.csv", usecols=["target_macro_gas_kwh"])["target_macro_gas_kwh"].values
EB = np.array([0.6, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 9.0]) * 1e6
ch, ct = np.histogram(pH["gas"], EB)[0], np.histogram(g_t, EB)[0]
b = np.digitize(pH["gas"], EB) - 1
ok = (b >= 0) & (b < len(EB) - 1)
w = np.full(len(H), (1 - ct.sum() / len(g_t)) / max((~ok).mean(), 1e-12))
w[ok] = ((ct / len(g_t)) / np.maximum(ch / len(H), 1e-12))[b[ok]]
w /= w.sum()
wt = {}
for rl in ("IMR", "S3"):
    wt[rl] = sum((np.sqrt(np.sum(w * se[rl][m])) - np.sqrt(np.sum(w * se0[m]))) / SIG[m] / 5 for m in yH)
print(f"importance-weighted (test mix): IMR {wt['IMR']:+.8f}  S3 {wt['S3']:+.8f}  IMR - S3 {wt['IMR'] - wt['S3']:+.8f}")
print(f"elapsed {time.time() - T0:.1f} s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
