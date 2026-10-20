"""Adoption lens, check 6: brute-force search for an exact / near-exact formula of the missing heating category.

Targets on labelled subzones: r = rest - adoption (percent), q = count_r / K (share of the non-listed households in the
missing category), cr = count_r (households).
Search space: the 113+3 subzone-constant columns (socio_demog_*, bld_*, env_*, hvac_dist_*) plus rest, N, K:
1. single columns and simple transforms (x, log1p x, sqrt x, x / rest, x * N / 100)
2. all pairs: x_i / x_j, x_i * x_j, x_i - x_j, x_i + x_j  (Pearson correlation with q, r, cr)
3. count matching: share of subzones with cr > 0 where round(x * N / 100) == cr or round(x * K / 100) == cr
An exact identity would show |corr| ~ 1 or an exact-match share ~ 1.
usage: .venv/bin/python exp/audit/adoption/a6_formula_search.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import HVAC  # noqa: E402

OUT = "exp/audit/adoption"
S = pd.read_parquet(f"{OUT}/sz.parquet")
const = json.load(open(f"{OUT}/const_cols.json"))["const_cols"]
L = S[S.ad.notna() & (S.K > 0)].copy()
cols = [c for c in const if L[c].std() > 0]
X = L[cols].values.astype(np.float64)
X = np.c_[X, L.rest.values, L.N.values, L.K.values]
names = cols + ["rest", "N", "K"]
targets = {"q": L.q.values, "r": L.r.values, "cr": L.cr.values}


def zs(a):
    a = np.where(np.isfinite(a), a, np.nan)
    m = np.nanmean(a, 0)
    a = np.where(np.isnan(a), m, a)
    s = a.std(0)
    s[s == 0] = np.inf
    return (a - a.mean(0)) / s


T = {k: zs(v[:, None])[:, 0] for k, v in targets.items()}
n = len(L)
best = []
Z = zs(X)
for k, t in T.items():
    c = Z.T @ t / n
    i = np.argmax(np.abs(c))
    best.append(("single", k, names[i], float(c[i])))
    for tf, f in (("log1p", np.log1p(np.maximum(X, 0))), ("sqrt", np.sqrt(np.maximum(X, 0))),
                  ("x/rest", X / np.maximum(L.rest.values[:, None], 1e-9)), ("x*N", X * L.N.values[:, None] / 100)):
        c = zs(f).T @ t / n
        i = np.nanargmax(np.abs(c))
        best.append((tf, k, names[i], float(c[i])))
print("best single-column correlations:")
for b in best:
    print("  ", b)
# postcode-level (EPC / meter) columns, averaged over the subzone's train + test rows
mcols = [c for c in S.columns if c.endswith("_m")]
XM = L[mcols].values.astype(np.float64)
for k, t in T.items():
    c = zs(XM).T @ t / n
    o = np.argsort(-np.abs(c))[:5]
    print(f"top postcode-mean columns vs {k}:", [(mcols[i], round(float(c[i]), 4)) for i in o])
    best.append(("pc_mean", k, mcols[o[0]], float(c[o[0]])))

# pairs
top = {k: [] for k in T}
p = X.shape[1]
for i in range(p):
    xi = X[:, [i]]
    ops = {"ratio": xi / np.where(X == 0, np.nan, X), "prod": xi * X, "diff": xi - X, "sum": xi + X}
    for op, M in ops.items():
        Mz = zs(M)
        for k, t in T.items():
            c = Mz.T @ t / n
            c[i] = 0
            j = int(np.nanargmax(np.abs(c)))
            top[k].append((abs(float(c[j])), float(c[j]), op, names[i], names[j]))
print("\nbest pair correlations (abs corr, corr, op, a, b):")
res = {}
for k in T:
    s = sorted(top[k], reverse=True)[:8]
    res[k] = s
    print(" target", k)
    for row in s:
        print("   ", tuple(round(x, 4) if isinstance(x, float) else x for x in row))

# count matching
pos = L.cr.values > 0
cr = L.cr.values[pos]
match = []
for j, nm in enumerate(names[:-3]):
    x = X[pos, j]
    mN = np.mean(np.round(x * L.N.values[pos] / 100) == cr)
    mK = np.mean(np.round(x * L.K.values[pos] / 100) == cr)
    match.append((max(mN, mK), nm, float(mN), float(mK)))
match.sort(reverse=True)
print("\nexact count matches on subzones with cr > 0 (n=%d): best" % pos.sum())
for m in match[:8]:
    print("   ", m)
# chance reference: cr distribution mode share
print("   chance ref: share of the most common cr value among cr>0:", float(pd.Series(cr).value_counts(normalize=True).iloc[0]))
json.dump(dict(single=best, pairs=res, count_match=match[:20]), open(f"{OUT}/a6_formula_search.json", "w"), indent=1)
