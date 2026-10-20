"""Light check: hard bounds of the scored targets in train, and how often the submitted predictions (sub_v19vce)
and the honest holdout-chain predictions (chain_H.npz) fall outside them."""
import numpy as np
import pandas as pd

T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
cols = T + ["bld_dim_06", "bld_dim_12", "bld_dim_01"]
tr = pd.read_parquet("cache/train.parquet", columns=cols)
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12", "bld_dim_01"])
sub = pd.read_csv("submissions/sub_v19vce.csv")
A, M = tr.bld_dim_06.values, tr.bld_dim_12.values.astype(float)
for c in T:
    y = tr[c].values
    print(c, "min", y.min(), "max", y.max(), "q(1e-5,1-1e-5)", np.quantile(y, [1e-5, 1 - 1e-5]))
g, e, u = (tr[c].values for c in T)
print("gas per meter min/max", (g / M).min(), (g / M).max(), "quantiles", np.quantile(g / M, [1e-4, 1e-3, 0.999, 0.9999]))
print("elec per meter min/max", (e / M).min(), (e / M).max(), "quantiles", np.quantile(e / M, [1e-4, 1e-3, 0.999, 0.9999]))
print("eui min/max", u.min(), u.max())
print("A min/max", A.min(), A.max(), "M min/max", M.min(), M.max())
# top of the gas distribution
gs = np.sort(g)[::-1]
print("largest 10 gas", gs[:10])
print("smallest 10 gas", np.sort(g)[:10])
print("largest 10 elec", np.sort(e)[::-1][:10])
print("smallest 10 elec", np.sort(e)[:10])
# test predictions vs bounds
At = te.bld_dim_06.values
Mt = te.bld_dim_12.values.astype(float)
sg, se, su = (sub[c].values for c in T)
print("sub gas > train max:", int((sg > g.max()).sum()), "max pred", sg.max())
print("sub elec > train max:", int((se > e.max()).sum()), "max pred", se.max())
print("sub eui > train max:", int((su > u.max()).sum()), "max pred", su.max())
print("sub gas < train min:", int((sg < g.min()).sum()), "sub elec < train min:", int((se < e.min()).sum()), "sub eui < train min:", int((su < u.min()).sum()))
print("sub eui > gasmax/A:", int((su > g.max() / At).sum()))
print("sub gas/M > train max gas/M:", int((sg / Mt > (g / M).max()).sum()), " < min:", int((sg / Mt < (g / M).min()).sum()))
print("sub elec/M > train max:", int((se / Mt > (e / M).max()).sum()), " < min:", int((se / Mt < (e / M).min()).sum()))
print("sub eui*A vs gas rel diff: median", np.median(np.abs(su * At - sg) / sg), "p99", np.quantile(np.abs(su * At - sg) / sg, 0.99))
