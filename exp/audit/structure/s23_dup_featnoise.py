"""K0 duplicate records have identical targets but differ in ~20 bld_dim columns. For each column: share of pairs equal and
partner correlation (dup pairs vs same-subzone random pairs). Columns with ~0 partner agreement are re-drawn per record."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
names = pq.read_schema("cache/train.parquet").names
B = sorted(c for c in names if c.startswith(("bld_dim", "bld_era")) and c not in K0 and c != "bld_dim_18")
tr = pd.read_parquet("cache/train.parquet", columns=K0 + B)
k = pd.util.hash_pandas_object(tr[K0], index=False).values
s = pd.Series(k); idx = np.where(s.map(s.value_counts()).values == 2)[0]
idx = idx[np.argsort(k[idx], kind="stable")]; a, b = idx[0::2], idx[1::2]
# random same-subzone pairs (different records)
sz = tr.groupby("subzone_id").indices
rng = np.random.default_rng(0); ra, rb = [], []
for v in sz.values():
    if len(v) >= 2:
        i, j = rng.choice(v, 2, replace=False); ra.append(i); rb.append(j)
ra, rb = np.array(ra[:20000]), np.array(rb[:20000])
rows = []
for c in B:
    x = tr[c].values
    eq = np.mean(x[a] == x[b]); eqr = np.mean(x[ra] == x[rb])
    cc = np.corrcoef(x[a], x[b])[0, 1] if x[a].std() > 0 else np.nan
    ccr = np.corrcoef(x[ra], x[rb])[0, 1] if x[ra].std() > 0 else np.nan
    rows.append((c, round(eq, 3), round(eqr, 3), round(cc, 3), round(ccr, 3)))
df = pd.DataFrame(rows, columns=["col", "eq_dup", "eq_samesub", "corr_dup", "corr_samesub"]).sort_values("eq_dup")
df.to_csv("exp/audit/structure/s23_dup_featnoise.csv", index=False)
print("dup pairs", len(a)); print(df.head(25).to_string(index=False))
