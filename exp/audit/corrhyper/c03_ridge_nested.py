"""corrhyper c03: nested check of a ridge penalty on the correction weights (c02 showed ridge < ols at every alpha, best
at the grid edge). Outer 5-fold on H = the chain folds (KFold(5, shuffle, rs=0)); in each outer-train part an inner
5-fold picks alpha from a grid, then lambda is refit on the outer-train part with that alpha and applied to the
outer-test part. Fit sizes match the chain (~103k rows), so this is the honest analogue of 'choose alpha on H, fit
lambda on all of H, apply to test'. Also prints the fixed-alpha curve (all-H 5-fold) over a wider grid, and NNLS+ridge.
"""
import json

import numpy as np
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
logf = open("exp/audit/corrhyper/c03_ridge_nested.log", "w")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def ridge(Fa, ra, a):
    s = np.sqrt((Fa ** 2).mean(0)) + 1e-30
    X = Fa / s
    G = X.T @ X
    return np.linalg.solve(G + a * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), X.T @ ra) / s


def cvpred(F, r, a, seed=0):
    pc = np.zeros(len(r))
    for tr_, te_ in KFold(5, shuffle=True, random_state=seed).split(F):
        pc[te_] = F[te_] @ ridge(F[tr_], r[tr_], a)
    return pc


OUT, tot_n, tot_f = {}, 0.0, 0.0
for m in ("gas", "eui", "elec"):
    F = z[f"{m}_F"].astype(np.float64)
    r = z[f"{m}_y"] - z[f"{m}_teacher"]
    sig = SIG[m]
    curve = {a: np.sqrt(np.mean((r - cvpred(F, r, a)) ** 2)) / sig for a in GRID}
    log(f"== {m}: fixed-alpha all-H curve (vs ols): " + ", ".join(f"{a}:{curve[a] - curve[0.0]:+.7f}" for a in GRID))
    nested, picks = np.zeros(len(r)), []
    for tr_, te_ in KFold(5, shuffle=True, random_state=0).split(F):
        Fi, ri = F[tr_], r[tr_]
        inner = {a: np.mean((ri - cvpred(Fi, ri, a, seed=1)) ** 2) for a in GRID}
        a = min(inner, key=inner.get)
        picks.append(a)
        nested[te_] = F[te_] @ ridge(Fi, ri, a)
    sn = np.sqrt(np.mean((r - nested) ** 2)) / sig
    # per-fold sign of the gain
    ols = cvpred(F, r, 0.0)
    folds = [float(np.sqrt(np.mean((r[te_] - nested[te_]) ** 2)) - np.sqrt(np.mean((r[te_] - ols[te_]) ** 2))) / sig
             for _, te_ in KFold(5, shuffle=True, random_state=0).split(F)]
    log(f"  nested ridge: {sn:.6f} vs ols {curve[0.0]:.6f} ({sn - curve[0.0]:+.7f}); alpha picks {picks}; "
        f"per-fold diffs {np.round(folds, 7).tolist()}")
    # lambda shift at the typical pick, fitted on all H
    a = float(np.median(picks))
    l0, la = ridge(F, r, 0.0), ridge(F, r, a)
    log(f"  lambda all-H ols {np.round(l0, 3).tolist()}")
    log(f"  lambda all-H a={a} {np.round(la, 3).tolist()}")
    rms0, rmsa = np.sqrt(np.mean((F @ l0) ** 2)) / sig, np.sqrt(np.mean((F @ la) ** 2)) / sig
    log(f"  correction rms/sigma ols {rms0:.5f}, ridge {rmsa:.5f}; corr {np.corrcoef(F @ l0, F @ la)[0, 1]:.5f}")
    OUT[m] = dict(curve={str(k): float(v - curve[0.0]) for k, v in curve.items()}, nested=float(sn - curve[0.0]),
                  picks=picks, folds=folds)
    tot_n += (sn - curve[0.0]) / 5
log(f"TOTAL nested ridge on lambda: {tot_n:+.8f}")
OUT["total_nested"] = tot_n
json.dump(OUT, open("exp/audit/corrhyper/c03_ridge_nested.json", "w"), indent=1)
