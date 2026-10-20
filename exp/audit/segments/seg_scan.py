"""Broad scan: residual bias of the honest H predictions (chain_H.npz m_corrected) by deciles of every raw feature.

For each feature (numeric; strings as category codes; <= 10 distinct values used as-is) and metric (gas, eui, elec):
sum of squared bin t-stats vs its df (chi-square excess = systematic bias), and the cross-fitted gain of a per-bin
shift (5-fold x 3 reps on H). Columns are read in chunks to keep memory low.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_scan.py > exp/audit/segments/seg_scan.log
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import chi2
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H = z["H"]
E = {m: z[f"{m}_y"] - z[f"{m}_corrected"] for m in SIG}
BASE = {m: np.sqrt(np.mean(E[m] ** 2)) / SIG[m] for m in SIG}
FOLDS = [list(KFold(5, shuffle=True, random_state=r).split(H)) for r in range(3)]
names = [n for n in pq.read_schema("cache/train.parquet").names if n not in ("id",) and not n.startswith("target_")]


def bins_of(x):
    x = pd.Series(x)
    if x.dtype == object or str(x.dtype).startswith("large") or x.dtype.name == "string":
        x = x.astype("category").cat.codes
    u = x.nunique()
    if u <= 10:
        return x.rank(method="dense").values.astype(int) - 1
    r = x.rank(method="first").values
    return np.minimum((r - 1) * 10 // len(x), 9).astype(int)


def stats(g, e, m):
    df = pd.DataFrame({"g": g, "e": e}).groupby("g")["e"].agg(["size", "mean", "std"])
    df = df[df["size"] > 1]
    t2 = float(((df["mean"] / (df["std"] / np.sqrt(df["size"]))) ** 2).sum())
    k = len(df)
    outs = []
    for folds in FOLDS:
        en = e.copy()
        for a, b in folds:
            c = pd.Series(e[a]).groupby(g[a]).mean()
            en[b] = e[b] - pd.Series(g[b]).map(c).fillna(0).values
        outs.append((np.sqrt(np.mean(en ** 2)) / SIG[m] - BASE[m]) / 5)
    return t2, k, float(np.mean(outs))


rows = []
pf = pq.ParquetFile("cache/train.parquet")
for i in range(0, len(names), 25):
    cols = names[i:i + 25]
    X = pf.read(columns=cols).to_pandas()
    X = X.iloc[H].reset_index(drop=True)
    for c in cols:
        g = bins_of(X[c].values)
        for m in SIG:
            t2, k, cf = stats(g, E[m], m)
            rows.append(dict(feature=c, metric=m, t2=t2, df=k, p=float(chi2.sf(t2, k)), cf=cf))
    del X
    print("done", i + len(cols), "features", flush=True)
R = pd.DataFrame(rows)
R.to_csv("exp/audit/segments/seg_scan.csv", index=False)
for m in SIG:
    r = R[R.metric == m].sort_values("cf")
    print(f"\n=== {m}: {len(r)} features; chi-square p < 0.001: {int((r.p < 1e-3).sum())}; p < 0.01: {int((r.p < 1e-2).sum())}"
          f"; cross-fitted shift gain < 0: {int((r.cf < 0).sum())}; min cf {r.cf.min():+.8f}")
    for _, x in r.head(8).iterrows():
        print(f"   {x.feature:18s} t2 {x.t2:7.1f} df {int(x.df):2d} p {x.p:.2e} cf dTotal {x.cf:+.8f}")
    print("   smallest chi-square p:")
    for _, x in r.sort_values("p").head(5).iterrows():
        print(f"   {x.feature:18s} t2 {x.t2:7.1f} df {int(x.df):2d} p {x.p:.2e} cf dTotal {x.cf:+.8f}")
