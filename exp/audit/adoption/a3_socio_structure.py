"""Adoption lens, check 3: structure of the subzone-constant census columns and their link to the missing category.

For each subzone-constant column f (labelled subzones):
- share of subzones where f * N / 100 is integral (same household denominator as hvac_dist)
- Pearson / Spearman correlation with r = rest - adoption, q = count_r / K, count_r, adoption
Also: greedy search for groups of columns that sum to 100 (census tables), and an OLS check of how much of r the
household-denominator counts explain (in-sample upper bound only, to see whether an exact identity can exist).
usage: .venv/bin/python exp/audit/adoption/a3_socio_structure.py
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
L = S[S.ad.notna()].copy()
cols = [c for c in const if c not in HVAC]
rows = []
N = L.N.values
for c in cols:
    f = L[c].values.astype(float)
    x = f * N / 100
    integ = np.mean(np.abs(x - np.round(x)) < 1e-6 * np.maximum(1, N / 100))
    fint = np.mean(np.abs(f - np.round(f)) < 1e-9)
    rk = pd.Series(f).rank().values
    rr = {}
    for t in ("r", "q", "cr", "ad", "rest"):
        y = L[t].values.astype(float)
        m = np.isfinite(y) & np.isfinite(f)
        rr["pc_" + t] = np.corrcoef(f[m], y[m])[0, 1] if f[m].std() > 0 else np.nan
        rr["sc_" + t] = np.corrcoef(rk[m], pd.Series(y[m]).rank().values)[0, 1] if f[m].std() > 0 else np.nan
    rows.append(dict(col=c, mean=f.mean(), min=f.min(), max=f.max(), frac_int_value=fint, frac_int_with_N=integ, **rr))
T = pd.DataFrame(rows).sort_values("frac_int_with_N", ascending=False)
T.to_csv(f"{OUT}/a3_columns.csv", index=False)
pd.set_option("display.width", 250)
print(T[["col", "mean", "max", "frac_int_value", "frac_int_with_N", "pc_r", "pc_q", "sc_q", "pc_ad"]]
      .round(4).to_string(index=False))
print("\nTop |corr| with q:")
print(T.reindex(T.pc_q.abs().sort_values(ascending=False).index)[["col", "pc_q", "sc_q", "pc_r", "pc_cr"]]
      .head(15).round(4).to_string(index=False))

# groups summing to 100 among household-denominator columns (greedy on contiguous runs of column order)
hh = T[T.frac_int_with_N > 0.98].col.tolist()
order = [c for c in const if c in hh]
print("\nhousehold-denominator columns:", len(order))
found = []
i = 0
vals = L[order].values
while i < len(order):
    s = np.zeros(len(L))
    for j in range(i, len(order)):
        s = s + vals[:, j]
        if np.mean(np.abs(s - 100) < 1e-6) > 0.99:
            found.append((order[i], order[j], j - i + 1))
            break
    i += 1
print("contiguous runs summing to 100 (start, end, n):", found)
json.dump(dict(household_cols=order, runs=found), open(f"{OUT}/a3_structure.json", "w"), indent=1)
