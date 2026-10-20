"""Adoption lens, check 5: what the unseen-subzone adoption still costs, and cheap honest post-hoc fixes.

Inputs: the saved adopt_v2-equivalent out-of-fold predictions (exp/adopt/base_{q_xent,r_reg}_{oof,unseen}.npy from
src/exp_adopt_v3.py base, byte-identical to adopt_v2 per JOURNAL 2026-10-19) and exp/audit/adoption/sz.parquet.
1. alignment check: reproduce CV unseen-weighted RMSE 1.0303 and the unseen predictions in exp/pred/adopt_v2.npy
2. cost: test adoption NRMSE and its share of the total score
3. noise decomposition: E[e^2 | K] = a K + b K^2 in count units (binomial part vs between-subzone q error)
4. calibration by K, nrow, rest; honest cross-fitted post-hoc corrections (5-fold over labelled subzones,
   fitted and scored with the unseen row-mix weights): scale of r, q-space affine, mix weight q_xent vs r_reg,
   per-nrow scale, shrink towards neighbour pooled q
usage: .venv/bin/python exp/audit/adoption/a5_oof_analysis.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from adopt_v2 import UNSEEN_MIX  # noqa: E402

OUT = "exp/audit/adoption"
SIG = 7.18
S = pd.read_parquet(f"{OUT}/sz.parquet")
tr_sz = pd.read_parquet("cache/train.parquet", columns=["subzone_id"]).subzone_id
te_sz = pd.read_parquet("cache/test.parquet", columns=["subzone_id"]).subzone_id
L_order = pd.unique(tr_sz.values)
L_set = set(L_order)
U_order = [s for s in pd.unique(te_sz.values) if s not in L_set]
L = S.loc[L_order].copy()
U = S.loc[U_order].copy()
for m in ("q_xent", "r_reg"):
    L[m] = np.load(f"exp/adopt/base_{m}_oof.npy")
    U[m] = np.load(f"exp/adopt/base_{m}_unseen.npy")
L["r_hat"] = (L.q_xent + L.r_reg) / 2
U["r_hat"] = (U.q_xent + U.r_reg) / 2

# row-mix weights: each labelled subzone weighted so that nrow buckets match the unseen test row mix
bucket = np.minimum(L.nrow.values, 6)
cnt = pd.Series(bucket).value_counts()
w = np.array([UNSEEN_MIX[b] / cnt[b] for b in bucket])
w = w / w.sum()


def rmse_w(pred_ad, idx=None):
    if idx is None:
        idx = np.arange(len(L))
    e2 = (np.clip(pred_ad, 0, 100) - L.ad.values[idx]) ** 2
    return float(np.sqrt(np.sum(w[idx] * e2) / np.sum(w[idx])))


def adopt_from_r(r, D=L):
    return np.clip(D.rest.values - r, 0, 100)


res = {}
# 1. alignment
e2 = (adopt_from_r(L.r_hat.values) - L.ad.values) ** 2
byn = pd.Series(e2).groupby(bucket).mean()
res["cv_unseen_weighted_rmse_reproduced"] = float(np.sqrt(sum(UNSEEN_MIX[k] * byn[k] for k in UNSEEN_MIX)
                                                     / sum(UNSEEN_MIX.values())))
res["cv_unseen_weighted_rmse_wfun"] = rmse_w(adopt_from_r(L.r_hat.values))
te_ad = np.load("exp/pred/adopt_v2.npy")
seen = te_sz.isin(L_set).values
pu = pd.Series(adopt_from_r(U.r_hat.values, U), index=U.index)
res["unseen_pred_matches_adopt_v2_maxabs"] = float(np.abs(te_ad[~seen] - te_sz[~seen].map(pu).values).max())

# 2. cost
fu = float((~seen).mean())
rm = res["cv_unseen_weighted_rmse_wfun"]
res["frac_unseen_rows"] = fu
res["test_adoption_nrmse_est"] = float(np.sqrt(fu) * rm / SIG)
res["total_share_est"] = res["test_adoption_nrmse_est"] / 5
res["global_q_rmse"] = rmse_w(adopt_from_r(L.rest.values * (L.cr.sum() / L.K.sum())))

# 3. noise decomposition in count units, unseen-mix weighted by rows: e = cr - K q_hat (q_hat = r_hat/rest)
Kv, Nv = L.K.values.astype(float), L.N.values.astype(float)
qh = np.where(L.rest.values > 0, L.r_hat.values / np.maximum(L.rest.values, 1e-12), 0)
ec = L.cr.values - Kv * qh
m = Kv > 0
X = np.c_[Kv[m], Kv[m] ** 2]
coef, *_ = np.linalg.lstsq(X * np.sqrt(w[m])[:, None], ec[m] ** 2 * np.sqrt(w[m]), rcond=None)
res["E_e2_given_K_coef_aK_bK2"] = coef.tolist()
# implied floors (adoption units^2 = count^2 * (100/N)^2)
s2 = (100 / Nv) ** 2
mse_now = np.sum(w * (adopt_from_r(L.r_hat.values) - L.ad.values) ** 2)
binom_part = np.sum(w * coef[0] * Kv * s2)
res["mse_now"] = float(mse_now)
res["implied_binomial_floor_rmse"] = float(np.sqrt(binom_part))
res["implied_reducible_share_of_mse"] = float(1 - binom_part / mse_now)
# direct binomial expectation with the model's own q_hat (if q_hat were the truth)
res["binomial_rmse_if_qhat_true"] = float(np.sqrt(np.sum(w * Kv * qh * (1 - qh) * s2)))
# R2 of r (row-mix weighted)
y = L.r.values
res["R2_r_weighted"] = float(1 - np.sum(w * (y - L.r_hat.values) ** 2) / np.sum(w * (y - np.sum(w * y)) ** 2))

# 4. calibration tables
L["w"] = w
L["e_r"] = L.r.values - L.r_hat.values
L["Kb"] = pd.cut(L.K, [-1, 0, 1, 2, 4, 8, 16, 32, 1e9])
L["nb"] = np.minimum(L.nrow, 6)
cal = {}
for key in ("Kb", "nb"):
    g = L.groupby(key, observed=True)
    cal[key] = pd.DataFrame({"n": g.size(), "wsum": g.w.sum(), "mean_r": g.r.mean(), "mean_rhat": g.r_hat.mean(),
                             "mean_err": g.e_r.mean(),
                             "rmse_ad": g.apply(lambda d: np.sqrt(np.mean((np.clip(d.rest - d.r_hat, 0, 100) - d.ad) ** 2)))})
    print(f"\ncalibration by {key}:\n", cal[key].round(4).to_string())

# honest cross-fitted post-hoc corrections
folds = list(KFold(5, shuffle=True, random_state=2026).split(L))
nbq = None


def crossfit(fit_apply):
    out = np.zeros(len(L))
    for a, b in folds:
        out[b] = fit_apply(a, b)
    return rmse_w(out)


def scale_r(a, b):
    x, y = L.r_hat.values, L.r.values
    k = np.sum(w[a] * x[a] * y[a]) / np.sum(w[a] * x[a] ** 2)
    return adopt_from_r(k * x[b], L.iloc[b])


def mixw(a, b):
    best, bw = 1e9, 0.5
    for t in np.linspace(0, 1, 21):
        rr = t * L.q_xent.values[a] + (1 - t) * L.r_reg.values[a]
        s = rmse_w(adopt_from_r(rr, L.iloc[a]), a)
        if s < best:
            best, bw = s, t
    return adopt_from_r(bw * L.q_xent.values[b] + (1 - bw) * L.r_reg.values[b], L.iloc[b])


def scale_by_n(a, b):
    out = np.zeros(len(b))
    x, y = L.r_hat.values, L.r.values
    for nb in range(1, 7):
        ma = bucket[a] == nb
        mb = bucket[b] == nb
        k = np.sum(w[a][ma] * x[a][ma] * y[a][ma]) / np.sum(w[a][ma] * x[a][ma] ** 2)
        out[mb] = adopt_from_r(k * x[b][mb], L.iloc[b][mb])
    return out


def affine_q(a, b):
    # r' = rest * clip(c0 + c1*qhat, 0, 1), fitted by weighted least squares on r
    rest = L.rest.values
    A = np.c_[rest, rest * qh]
    c, *_ = np.linalg.lstsq(A[a] * np.sqrt(w[a])[:, None], L.r.values[a] * np.sqrt(w[a]), rcond=None)
    qq = np.clip(c[0] + c[1] * qh[b], 0, 1)
    return adopt_from_r(rest[b] * qq, L.iloc[b])


hon = {"baseline": rmse_w(adopt_from_r(L.r_hat.values)), "scale_r": crossfit(scale_r), "mix_weight": crossfit(mixw),
       "scale_by_nrow": crossfit(scale_by_n), "affine_q": crossfit(affine_q)}
base = hon["baseline"]
res["honest_posthoc_rmse"] = hon
res["honest_posthoc_total_change"] = {k: float(np.sqrt(fu) * (v - base) / SIG / 5) for k, v in hon.items()}
for k, v in res.items():
    print(k, v)
json.dump(res, open(f"{OUT}/a5_oof_analysis.json", "w"), indent=1)
L[["K", "N", "rest", "cr", "r", "ad", "nrow", "r_hat", "q_xent", "r_reg", "w"]].to_parquet(f"{OUT}/L_oof.parquet")
U[["K", "N", "rest", "nrow", "r_hat", "q_xent", "r_reg"]].to_parquet(f"{OUT}/U_pred.parquet")
