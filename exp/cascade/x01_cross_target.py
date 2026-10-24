"""X01 (post-competition): does cross-target information help on top of the final pipeline?

Idea from the 1st-place writeup (opo yo): gas predictions feed the electricity model and electricity refines gas.
Tested on the honest holdout chain (exp/audit/chain_H.npz: models trained on P, blend predictions 'teacher' on H,
neighbour-residual features F built from P residuals only). Every variant is fitted with the same outer
KFold(5, shuffle, random_state=0) over H as the corrhyper audit, so all scores are out-of-fold on H.

Variants per scored column m (residual r = y - teacher):
  own_lin   OLS on F_m                                  = the final pipeline's residual correction
  cross_lin OLS on F_m + F of the other family          = cross-target residual correction
  own_gbm   LightGBM on teacher_m, F_m, log A, log M    = nonlinear second stage without cross-target input
  cross_gbm own_gbm + teacher and F of the other family = cascade / bidirectional refinement
Other family: elec -> gas and eui; gas and eui -> elec. LightGBM stops early on an inner 10% split of the
outer-train part, never on the outer-test part.
usage: python exp/cascade/x01_cross_target.py [--smoke]
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

SMOKE = "--smoke" in sys.argv
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
OTHER = {"gas": ["elec"], "eui": ["elec"], "elec": ["gas", "eui"]}
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, deterministic=True, force_row_wise=True,
              num_threads=8, verbose=-1)
OUT = "exp/cascade/x01_cross_target" + ("_smoke" if SMOKE else "")

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H = z["H"]
A, M = z["A"][H], z["M"][H]
n = len(H)
rows = np.arange(n)
if SMOKE:
    rows = np.random.default_rng(0).choice(n, 20000, replace=False)
logf = open(OUT + ".log", "w")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def get(m, key):
    return z[f"{m}_{key}"][rows].astype(np.float64)


def ols_cv(X, r, folds):
    out = np.zeros(len(r))
    for a, b in folds:
        lam, *_ = np.linalg.lstsq(X[a], r[a], rcond=None)
        out[b] = X[b] @ lam
    return out


def gbm_cv(X, r, folds):
    out, iters = np.zeros(len(r)), []
    for a, b in folds:
        inner = np.random.default_rng(1).random(len(a)) < 0.1
        fit, val = a[~inner], a[inner]
        da = lgb.Dataset(X[fit], r[fit])
        dv = lgb.Dataset(X[val], r[val], reference=da)
        mdl = lgb.train(PARAMS, da, 200 if SMOKE else 3000, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        out[b] = mdl.predict(X[b], num_iteration=mdl.best_iteration)
        iters.append(mdl.best_iteration)
    return out, iters


def nrmse(e, m):
    return float(np.sqrt(np.mean(e ** 2)) / SIG[m])


t0 = time.time()
folds = list(KFold(2 if SMOKE else 5, shuffle=True, random_state=0).split(rows))
res = {}
for m in ("gas", "eui", "elec"):
    y, teach, F = get(m, "y"), get(m, "teacher"), get(m, "F")
    r = y - teach
    Fo = np.hstack([get(o, "F") for o in OTHER[m]])
    To = np.column_stack([get(o, "teacher") for o in OTHER[m]])
    scale = np.column_stack([np.log(A[rows]), np.log(M[rows])])
    preds = {"none": np.zeros(n if not SMOKE else len(rows)),
             "own_lin": ols_cv(F, r, folds),
             "cross_lin": ols_cv(np.hstack([F, Fo]), r, folds)}
    preds["own_gbm"], it_own = gbm_cv(np.hstack([teach[:, None], F, scale]), r, folds)
    preds["cross_gbm"], it_cross = gbm_cv(np.hstack([teach[:, None], F, scale, To, Fo]), r, folds)
    score = {k: nrmse(r - p, m) for k, p in preds.items()}
    base = preds["own_lin"]
    per_fold = {k: [nrmse(r[b] - p[b], m) - nrmse(r[b] - base[b], m) for _, b in folds]
                for k, p in preds.items() if k != "own_lin"}
    res[m] = dict(score=score, delta_vs_pipeline={k: v - score["own_lin"] for k, v in score.items()},
                  per_fold_delta=per_fold, gbm_iters={"own": it_own, "cross": it_cross})
    log(f"== {m} ({time.time() - t0:.0f}s): " + ", ".join(f"{k} {v:.6f}" for k, v in score.items()))
    log("   delta vs pipeline correction: " + ", ".join(f"{k} {v:+.6f}" for k, v in res[m]["delta_vs_pipeline"].items()))
    log("   per-fold delta: " + "; ".join(f"{k} {np.round(v, 6).tolist()}" for k, v in per_fold.items()))

total = {k: sum(res[m]["delta_vs_pipeline"][k] for m in res) / 5 for k in ("cross_lin", "own_gbm", "cross_gbm")}
res["total_delta_on_5_target_mean"] = total
log("TOTAL change of the competition score (mean of 5 targets) vs pipeline correction: "
    + ", ".join(f"{k} {v:+.7f}" for k, v in total.items()))
json.dump(res, open(OUT + ".json", "w"), indent=1)
