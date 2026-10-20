"""verify_postproc/adv v01: independent re-derivation (own code, nothing imported from the auditor's scripts) of the
claimed S1+S3+S4 post-processing gain on the holdout H (exp/audit/chain_H.npz), plus label / index sanity checks.

S1  p *= 1 + d_g, g in {M in [2,3], M == 4}; d_g = sum(pP rP) / sum(pP^2) over the P rows of g (P-OOF teacher)
S3  (gas, eui) p += k * (mean shift of N(p, s^2) truncated above at U); s = p sqrt(a0 + b0 / M) with (a0, b0) from P
    residuals; U = max P gas (gas) or max P gas / A (eui); k = least squares of (y - p) on that shift, fit rows of H only
S4  clip to bounds from P: gas in [min eui * A, max gas], eui in [min eui, max gas / A], elec in [min, max](elec / A) * A
Every H-dependent number (only k) is fitted on 4/5 of H and applied to the other 1/5 (KFold, several seeds).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/adv/v01_rederive.py
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[:3]
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/adv"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
nH = len(H)

# ---------------------------------------------------------------- sanity: indices and labels vs the raw train table
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EL, EUI, "bld_dim_06", "bld_dim_12"])
assert len(np.intersect1d(P, H)) == 0 and len(P) + len(H) == len(tr) and len(np.union1d(P, H)) == len(tr)
chk = {"A==bld_dim_06": bool(np.array_equal(A, tr["bld_dim_06"].values)),
       "M==bld_dim_12": bool(np.array_equal(M, tr["bld_dim_12"].values.astype(float)))}
for m in MET:
    yt = tr[COL[m]].values
    chk[f"{m}_y==train[H]"] = float(np.max(np.abs(z[f"{m}_y"] - yt[H]) / SIGMAS[COL[m]]))
    chk[f"{m}_pP+rP==train[P]"] = float(np.max(np.abs(z[f"{m}_pP"] + z[f"{m}_rP"] - yt[P]) / SIGMAS[COL[m]]))
    # is pP an out-of-fold quantity? An in-sample fit would show a far smaller P residual than the H teacher residual
    chk[f"{m}_nrmse_P_teacher"] = float(np.sqrt(np.mean(z[f"{m}_rP"] ** 2)) / SIGMAS[COL[m]])
    chk[f"{m}_nrmse_H_teacher"] = float(np.sqrt(np.mean((z[f"{m}_y"] - z[f"{m}_teacher"]) ** 2)) / SIGMAS[COL[m]])
    chk[f"{m}_nrmse_H_corrected"] = float(np.sqrt(np.mean((z[f"{m}_y"] - z[f"{m}_corrected"]) ** 2)) / SIGMAS[COL[m]])
print("sanity", json.dumps(chk, indent=0))
del tr

AP, AH, MP, MH = A[P], A[H], M[P], M[H]
y = {m: z[f"{m}_y"] for m in MET}
pP = {m: z[f"{m}_pP"] for m in MET}
rP = {m: z[f"{m}_rP"] for m in MET}
yP = {m: pP[m] + rP[m] for m in MET}
base = {m: np.maximum(z[f"{m}_corrected"], 0.0) for m in MET}
gmax = yP["gas"].max()
euimin = yP["eui"].min()
ea = yP["elec"] / AP
lo_hi = {"gas": (euimin * AH, np.full(nH, gmax)), "eui": (np.full(nH, euimin), gmax / AH),
         "elec": (ea.min() * AH, ea.max() * AH)}
GROUPS = ((2, 3), (4, 4))
delta = {m: [float(pP[m][g] @ rP[m][g] / (pP[m][g] @ pP[m][g])) for g in ((MP >= a) & (MP <= b) for a, b in GROUPS)]
         for m in MET}
var_ab = {}
for m in ("gas", "eui"):
    X = np.column_stack([np.ones(len(P)), 1.0 / MP])
    var_ab[m] = np.linalg.lstsq(X, (rP[m] / pP[m]) ** 2, rcond=None)[0]


def trunc_shift(p, s, U):
    """E[Y] - p for Y ~ N(p, s^2) truncated to Y <= U: -s phi(b) / Phi(b), b = (U - p) / s (log-space for stability)"""
    b = (U - p) / s
    return -s * np.exp(norm.logpdf(b) - log_ndtr(b))


def s1(p, m):
    q = p.copy()
    for (a, b), d in zip(GROUPS, delta[m]):
        g = (MH >= a) & (MH <= b)
        q[g] = q[g] * (1.0 + d)
    return q


def s3_x(p, m):
    s = p * np.sqrt(np.maximum(var_ab[m][0] + var_ab[m][1] / MH, 1e-8))
    return trunc_shift(p, s, lo_hi[m][1])


def nr(m, idx, p):
    return float(np.sqrt(np.mean((y[m][idx] - p[idx]) ** 2)) / SIGMAS[COL[m]])


# S1 does not depend on H labels -> computed once; S3's x depends on predictions only, k on H labels (fit rows)
after1 = {m: s1(base[m], m) for m in MET}
xs3 = {m: s3_x(after1[m], m) for m in ("gas", "eui")}


def run(seed, steps=("S1", "S3", "S4")):
    out = {m: np.empty(nH) for m in MET}
    ks = {m: [] for m in ("gas", "eui")}
    for fit, app in KFold(5, shuffle=True, random_state=seed).split(np.arange(nH)):
        for m in MET:
            p = after1[m].copy() if "S1" in steps else base[m].copy()
            if "S3" in steps and m in xs3:
                x = xs3[m] if "S1" in steps else s3_x(base[m], m)
                k = float(x[fit] @ (y[m] - p)[fit] / (x[fit] @ x[fit]))
                ks[m].append(round(k, 4))
                p = p + k * x
            if "S4" in steps:
                p = np.clip(p, *lo_hi[m])
            out[m][app] = p[app]
    return out, ks


idx = np.arange(nH)
s0 = {m: nr(m, idx, base[m]) for m in MET}
res = {"sanity": chk, "deltas_P": delta, "var_ab": {m: v.tolist() for m, v in var_ab.items()},
       "bounds": dict(gmax=gmax, euimin=euimin, eamin=ea.min(), eamax=ea.max()), "S0": s0}
print("P-OOF deltas", {m: np.round(v, 4).tolist() for m, v in delta.items()}, "| var model",
      {m: np.round(v, 5).tolist() for m, v in var_ab.items()})
print("S0 (H corrected, clip 0)", {m: round(v, 6) for m, v in s0.items()})
tot_seed = []
for seed in range(10):
    pr, ks = run(seed)
    d = {m: nr(m, idx, pr[m]) - s0[m] for m in MET}
    tot = sum(d.values()) / 5
    tot_seed.append(tot)
    if seed == 0:
        res["seed0"] = dict(d=d, total=tot, k=ks)
        print("seed 0  S1+S3+S4:", {m: f"{v:+.7f}" for m, v in d.items()}, f"TOTAL {tot:+.8f}", "k", ks)
        # per-fold totals (fold-local NRMSE differences, same as the claim's per-fold numbers)
        ft = []
        for fit, app in KFold(5, shuffle=True, random_state=0).split(idx):
            ft.append(sum(nr(m, app, pr[m]) - nr(m, app, base[m]) for m in MET) / 5)
        res["seed0"]["fold_totals"] = ft
        print("   per-fold totals", np.round(ft, 8))
        # step decomposition (cumulative, same folds)
        for steps in (("S1",), ("S1", "S3"), ("S3",), ("S4",), ("S1", "S4")):
            prs, _ = run(0, steps)
            dd = {m: nr(m, idx, prs[m]) - s0[m] for m in MET}
            res[f"steps_{'+'.join(steps)}"] = dict(d=dd, total=sum(dd.values()) / 5)
            print(f"   steps {'+'.join(steps):10s}", {m: f"{v:+.7f}" for m, v in dd.items()},
                  f"TOTAL {sum(dd.values()) / 5:+.8f}")
print(f"TOTAL over KFold seeds 0..9: mean {np.mean(tot_seed):+.8f} min {np.min(tot_seed):+.8f} max {np.max(tot_seed):+.8f}")
res["total_seeds"] = tot_seed
json.dump(res, open(f"{OUT}/v01_rederive.json", "w"), indent=1, default=float)
