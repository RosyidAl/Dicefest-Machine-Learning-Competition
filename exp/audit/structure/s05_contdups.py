"""For each numeric column: number of distinct values; for near-continuous columns, train groups sharing an exact value,
share of those groups with identical gas/elec/A/subzone, and test rows matching a train value."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
T = ["target_macro_gas_kwh", "target_macro_elec_kwh"]
names = pq.read_schema("cache/train.parquet").names
num = [c for c in names if c.startswith(("bld_dim", "socio_demog", "env_climate", "hvac"))]
rows = []
for chunk in [num[i:i + 40] for i in range(0, len(num), 40)]:
    tr = pd.read_parquet("cache/train.parquet", columns=chunk + T + ["subzone_id", "bld_dim_06"] if "bld_dim_06" not in chunk else chunk + T + ["subzone_id"])
    te = pd.read_parquet("cache/test.parquet", columns=chunk)
    for c in chunk:
        if tr[c].dtype == object or str(tr[c].dtype).startswith("str"):
            continue
        nu = tr[c].nunique()
        if nu < 0.5 * len(tr):
            rows.append((c, nu, np.nan, np.nan, np.nan, np.nan, np.nan)); continue
        vc = tr[c].value_counts(); rep = vc[vc > 1]
        sub = tr[tr[c].isin(rep.index)]
        g = sub.groupby(c)
        same_gas = (g[T[0]].nunique() == 1).mean(); same_sub = (g["subzone_id"].nunique() == 1).mean(); same_A = (g["bld_dim_06"].nunique() == 1).mean()
        tm = te[c].isin(vc.index).sum()
        rows.append((c, nu, len(rep), round(same_gas, 3), round(same_sub, 3), round(same_A, 3), int(tm)))
        print(rows[-1], "top repeated", rep.head(3).to_dict(), flush=True)
df = pd.DataFrame(rows, columns=["col", "nunique", "rep_groups", "same_gas", "same_sub", "same_A", "test_match"])
df.to_csv("exp/audit/structure/s05_contdups.csv", index=False)
print(df[df.rep_groups.notna()].to_string())
