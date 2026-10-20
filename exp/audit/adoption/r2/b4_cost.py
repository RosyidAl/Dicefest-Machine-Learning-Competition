"""Adoption lens (round 2), check b4: what adoption still costs and what any fix could gain.

Inputs (read only): exp/audit/adoption/sz.parquet (subzone table, float64 census values), the production-equivalent
out-of-fold r predictions exp/adopt/base_{q_xent,r_reg}_oof.npy (src/exp_adopt_v3.py base == adopt_v2), the test
subzone ids (one column of cache/test.parquet) and the train subzone ids (one column of cache/train.parquet).
 1. test adoption NRMSE estimate = sqrt(frac_unseen * MSE_unseen_weighted) / 7.18, and its share of the total
 2. noise decomposition in count space: E[e^2 | K] = a K + b K (K - 1) (binomial part with mean q(1-q) = a, and
    between-subzone q error b = Var(q - q_hat)), fitted on labelled subzones with the unseen-mix weights
 3. floors: (i) q known exactly per subzone (pure binomial), (ii) global pooled q (no model), (iii) exact formula (0)
    -> total-score change if the unseen-subzone predictions reached each floor
 4. cheap post-hoc calibrations, 5-fold cross-fitted over labelled subzones with the unseen-mix weights:
    r' = s * r_hat;  r' = s_b * r_hat per nrow bucket;  convex mix of the two production models
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/adoption/r2/b4_cost.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from adopt_v2 import UNSEEN_MIX  # noqa: E402

OUT = "exp/audit/adoption/r2"
SIG, NT = 7.18, 183987
S = pd.read_parquet("exp/audit/adoption/sz.parquet")
tr_sz = pd.read_parquet("cache/train.parquet", columns=["subzone_id"]).subzone_id
te_sz = pd.read_parquet("cache/test.parquet", columns=["subzone_id"]).subzone_id
L = S.loc[pd.unique(tr_sz.values)].copy()
for m in ("q_xent", "r_reg"):
    L[m] = np.load(f"exp/adopt/base_{m}_oof.npy")
L["r_hat"] = (L.q_xent + L.r_reg) / 2
frac_unseen = float((~te_sz.isin(set(L.index))).mean())
b = np.minimum(L.nrow.values, 6)
cnt = pd.Series(b).value_counts()
w = np.array([UNSEEN_MIX[k] / cnt[k] for k in b])
w = w / w.sum()
res = {"frac_unseen_test_rows": frac_unseen}


def mse_w(r_pred, idx=slice(None)):
    e2 = (np.clip(L.rest.values[idx] - r_pred, 0, 100) - L.ad.values[idx]) ** 2
    return float(np.sum(w[idx] * e2) / np.sum(w[idx]))


def total_from_mse(m):
    return float(np.sqrt(frac_unseen * m) / SIG / 5)


m0 = mse_w(L.r_hat.values)
res["unseen_weighted_rmse_production"] = np.sqrt(m0)
res["test_adoption_nrmse_est"] = float(np.sqrt(frac_unseen * m0) / SIG)
res["adoption_share_of_total"] = total_from_mse(m0)

# 2. noise decomposition in count space (households): e = cr - K q_hat
N, K = L.N.values.astype(float), L.K.values.astype(float)
q_hat = np.clip(L.r_hat.values * N / 100 / np.maximum(K, 1), 0, 1)
e2c = (L.cr.values - K * q_hat) ** 2
m = K > 0
A_ = np.c_[K[m], K[m] * (K[m] - 1)]
coef = np.linalg.lstsq(A_ * np.sqrt(w[m])[:, None], e2c[m] * np.sqrt(w[m]), rcond=None)[0]
res["count_noise_fit_a_binomial"], res["count_noise_fit_b_qerror_var"] = (float(c) for c in coef)
res["mean_qhat_1mqhat_w"] = float(np.sum(w[m] * q_hat[m] * (1 - q_hat[m])) / np.sum(w[m]))
# 3. floors in percent space, unseen-mix weighted
s2 = (100 / np.maximum(N, 1)) ** 2
floor_q_exact = float(np.sum(w * s2 * K * coef[0]) / np.sum(w))  # binomial part only, q known per subzone
floor_plugin = float(np.sum(w * s2 * K * q_hat * (1 - q_hat)) / np.sum(w))
qg = L.cr[m].sum() / L.K[m].sum()
r_global = L.rest.values * qg
res["pooled_q"] = float(qg)
res["unseen_weighted_rmse_global_q"] = np.sqrt(mse_w(r_global))
res["floor_rmse_q_exact_fit"] = np.sqrt(floor_q_exact)
res["floor_rmse_q_exact_plugin"] = np.sqrt(floor_plugin)
for nm, f in (("q_exact_fit", floor_q_exact), ("q_exact_plugin", floor_plugin), ("exact_formula", 0.0)):
    res[f"total_change_if_{nm}"] = total_from_mse(f) - total_from_mse(m0)
# 4. cross-fitted post-hoc calibration
kf = KFold(5, shuffle=True, random_state=1)
idx_all = np.arange(len(L))
r_hat, rq, rr = L.r_hat.values, L.q_xent.values, L.r_reg.values
grid_s = np.linspace(0.85, 1.15, 61)
grid_a = np.linspace(0, 1, 21)
cal = {k: np.zeros(len(L)) for k in ("scale", "scale_by_nrow", "mix")}
for a, bb in kf.split(idx_all):
    s = grid_s[np.argmin([mse_w(g * r_hat[a], a) for g in grid_s])]
    cal["scale"][bb] = s * r_hat[bb]
    for k in range(1, 7):
        aa, bk = a[b[a] == k], bb[b[bb] == k]
        sk = grid_s[np.argmin([mse_w(g * r_hat[aa], aa) for g in grid_s])]
        cal["scale_by_nrow"][bk] = sk * r_hat[bk]
    al = grid_a[np.argmin([mse_w(g * rq[a] + (1 - g) * rr[a], a) for g in grid_a])]
    cal["mix"][bb] = al * rq[bb] + (1 - al) * rr[bb]
for k, v in cal.items():
    res[f"calib_{k}_rmse"] = np.sqrt(mse_w(v))
    res[f"calib_{k}_total_change"] = total_from_mse(mse_w(v)) - total_from_mse(m0)
# where the unseen error sits (unseen-mix weighted share of squared error by K bucket)
e2 = (np.clip(L.rest.values - r_hat, 0, 100) - L.ad.values) ** 2
kb = pd.cut(K, [-1, 0, 5, 10, 20, 40, 1000])
res["sq_error_share_by_K"] = {str(i): round(float(v), 4) for i, v in
                              (pd.Series(w * e2).groupby(kb, observed=True).sum() / np.sum(w * e2)).items()}
json.dump(res, open(f"{OUT}/b4_cost.json", "w"), indent=1, default=float)
for k, v in res.items():
    print(k, round(v, 6) if isinstance(v, float) else v)
