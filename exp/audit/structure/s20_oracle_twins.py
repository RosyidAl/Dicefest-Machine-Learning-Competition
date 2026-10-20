"""Oracle upper bound (NOT achievable: partners are not identifiable) for H rows that share gas, gas/M or gas/E exactly
with another train row: NRMSE change if those rows were predicted perfectly; and the expected number of test rows with such a train partner."""
import numpy as np, pandas as pd
S = {"gas": 151750.04, "eui": 40.65}
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_12", "bld_dim_01", "target_macro_gas_kwh"])
g = tr.target_macro_gas_kwh.values; M = tr.bld_dim_12.values; E = tr.bld_dim_01.values
def coll(x):
    s = pd.Series(np.round(x, 9)); return s.map(s.value_counts()).values > 1
c = coll(g) | coll(g / M) | coll(g / E)
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]; cH = c[H]
tot = 0
for m in ["gas", "eui"]:
    y = d[f"{m}_y"]; p = d[f"{m}_corrected"]; r2 = (y - p) ** 2
    b = np.sqrt(r2.mean()) / S[m]; o = np.sqrt(np.where(cH, 0, r2).mean()) / S[m]; tot += (o - b) / 5
    print(f"{m}: H rows with a collision partner {cH.mean():.4f}, SSE share {r2[cH].sum() / r2.sum():.4f}, oracle NRMSE {b:.5f} -> {o:.5f}")
print(f"oracle total change {tot:+.5f}; expected test rows with a train partner ~{int(c.mean() * 183987)}")
