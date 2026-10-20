"""Adoption lens (round 2), check b2: hunt for an exact / near-exact formula of the missing heating category.

Data: exp/audit/adoption/sz.parquet (one row per subzone, written by a2_subzone_table.py; read only here).
Targets on labelled subzones with K > 0 (rest > 0):  r = rest - adoption (percent), q = count_r / K, cr = count_r,
and ad (adoption) for reference.
Search space U: the subzone-constant columns with variance (socio_demog_*, bld_dim_43, hvac_dist_*) + rest, N, K.
 1. single columns (raw, log1p, x / rest, x * N / 100) and postcode-level means (_m columns)
 2. all ordered pairs in percent space: x_i / x_j, x_i * x_j, x_i - x_j, x_i + x_j
 3. count space: c = x * N / 100;  c_i - c_j and c_i + c_j against cr; exact integer matches round(c_i - c_j) == cr
 4. multivariate ceilings: 5-fold CV OLS on all U columns (and U + _m columns) for r, q, cr
An exact formula shows |corr| ~ 1, an exact-match share ~ 1, or CV R^2 ~ 1.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/adoption/r2/b2_formula.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import HVAC  # noqa: E402

OUT = "exp/audit/adoption/r2"
S = pd.read_parquet("exp/audit/adoption/sz.parquet")
const = [c for c in json.load(open(f"{OUT}/b1_verify.json"))["const_train_test_cols"] if c != "zone_region_id"]
L = S[S.ad.notna() & (S.K > 0)].copy()
U = [c for c in const if L[c].astype(float).std() > 0]
X = np.c_[L[U].values.astype(np.float64), L.rest.values, L.N.values.astype(float), L.K.values.astype(float)]
names = U + ["rest", "N", "K"]
n, p = X.shape
print(f"labelled subzones with K>0: {n}; candidate columns: {p}", flush=True)
T = {"r": L.r.values, "q": L.q.values, "cr": L.cr.values, "ad": L.ad.values}
TZ = {k: (v - v.mean()) / v.std() for k, v in T.items()}
res = {"n": n, "p": p}


def zcols(M):
    M = np.where(np.isfinite(M), M, np.nan)
    mu = np.nanmean(M, 0)
    M = np.where(np.isnan(M), mu, M)
    sd = M.std(0)
    sd[sd == 0] = np.inf
    return (M - M.mean(0)) / sd


def corr_all(M):
    Z = zcols(M)
    return {k: Z.T @ t / n for k, t in TZ.items()}


# 1. singles
single = {}
with np.errstate(all="ignore"):
    for tf, M in (("x", X), ("log1p", np.log1p(np.maximum(X, 0))), ("x/rest", X / L.rest.values[:, None]),
                  ("x*N", X * L.N.values[:, None] / 100)):
        c = corr_all(M)
        for k in T:
            o = np.argsort(-np.abs(np.nan_to_num(c[k])))[:3]
            single.setdefault(k, []).extend([(round(float(c[k][i]), 4), tf, names[i]) for i in o])
mcols = [c for c in S.columns if c.endswith("_m")]
cm = corr_all(L[mcols].values.astype(float))
for k in T:
    o = np.argsort(-np.abs(np.nan_to_num(cm[k])))[:3]
    single[k].extend([(round(float(cm[k][i]), 4), "pc_mean", mcols[i]) for i in o])
    single[k] = sorted(single[k], key=lambda z: -abs(z[0]))[:6]
res["single_top"] = single
print("best single-column |corr| per target:", {k: v[0] for k, v in single.items()}, flush=True)

# 2. pairs in percent space
top = {k: [] for k in T}
with np.errstate(all="ignore"):
    for i in range(p):
        xi = X[:, [i]]
        for op, M in (("ratio", xi / np.where(X == 0, np.nan, X)), ("prod", xi * X), ("diff", xi - X), ("sum", xi + X)):
            c = corr_all(M)
            for k in T:
                ck = np.nan_to_num(c[k])
                ck[i] = 0
                for j in np.argsort(-np.abs(ck))[:2]:
                    top[k].append((abs(float(ck[j])), round(float(ck[j]), 4), op, names[i], names[j]))
res["pair_top"] = {k: [z[1:] for z in sorted(v, reverse=True)[:8]] for k, v in top.items()}
for k in T:
    print(f"best pair for {k}:", res["pair_top"][k][:3], flush=True)

# 3. count space (household denominator N): c_i - c_j, c_i + c_j vs cr; exact integer matches
C = X[:, :len(U)] * L.N.values[:, None] / 100
cr = L.cr.values
crz = (cr - cr.mean()) / cr.std()
best_c, best_m = [], []
chance = float(pd.Series(cr).value_counts(normalize=True).iloc[0])
for i in range(len(U)):
    for op, M in (("cdiff", C[:, [i]] - C), ("csum", C[:, [i]] + C)):
        Z = zcols(M)
        cc = Z.T @ crz / n
        cc[i] = 0
        j = int(np.argmax(np.abs(cc)))
        best_c.append((abs(float(cc[j])), round(float(cc[j]), 4), op, U[i], U[j]))
        mt = (np.round(M) == cr[:, None]).mean(0)
        mt[i] = 0
        j = int(np.argmax(mt))
        best_m.append((float(mt[j]), op, U[i], U[j]))
    mt1 = float(np.mean(np.round(C[:, i]) == cr))
    best_m.append((mt1, "c", U[i], ""))
res["count_corr_top"] = [z[1:] for z in sorted(best_c, reverse=True)[:8]]
res["count_exact_match_top"] = sorted(best_m, reverse=True)[:8]
res["count_exact_match_chance_mode_share"] = chance
print("count-space best corr with cr:", res["count_corr_top"][:3])
print("exact integer match share best:", res["count_exact_match_top"][:3], "chance (mode share):", round(chance, 4),
      flush=True)

# 4. multivariate CV ceilings (ridge, standardised), percent space and with postcode means
Xs = zcols(X)
XM = np.c_[Xs, zcols(L[mcols].values.astype(float))]
kf = KFold(5, shuffle=True, random_state=0)
cv = {}
for nm, F in (("U", Xs), ("U+pc_mean", XM)):
    for k in ("r", "q", "cr"):
        y = T[k]
        oof = np.zeros(n)
        for a, b in kf.split(F):
            oof[b] = Ridge(alpha=1.0).fit(F[a], y[a]).predict(F[b])
        cv[f"{nm}:{k}"] = round(float(1 - np.mean((y - oof) ** 2) / y.var()), 4)
res["ridge_cv_r2"] = cv
print("ridge 5-fold CV R2:", cv, flush=True)
json.dump(res, open(f"{OUT}/b2_formula.json", "w"), indent=1, default=str)
