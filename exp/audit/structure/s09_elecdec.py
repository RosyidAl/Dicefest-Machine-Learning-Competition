"""Elec decimal structure: 1-decimal vs 3-decimal rows. Is elec/M round for the 3-decimal rows (elec = M x rounded mean)?
Is the class related to features (M, E, zone) and to honest residuals on H?"""
import numpy as np, pandas as pd
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_12", "bld_dim_01", "bld_dim_06", "zone_region_id", "target_macro_elec_kwh", "target_macro_gas_kwh"])
e = tr.target_macro_elec_kwh.values; M = tr.bld_dim_12.values; E = tr.bld_dim_01.values
def isround(x, d): return np.abs(x * 10**d - np.round(x * 10**d)) < 1e-4
c1 = isround(e, 1); c3 = ~c1
print("elec 1-dec share", c1.mean())
for d in [0, 1, 2, 3]:
    print(f"  3-dec rows: elec/M round at d{d}: {isround(e[c3] / M[c3], d).mean():.3f}   elec/E: {isround(e[c3] / E[c3], d).mean():.3f}  | 1-dec rows elec/M d{d}: {isround(e[c1] / M[c1], d).mean():.3f}")
# what multiplier? elec*k integer?
for k in [2, 3, 4, 5, 6, 8, 10, 20, 40, 1000]:
    print("  3-dec rows: elec*k/10 integer share k=", k, isround(e[c3] * k / 10, 0).mean().round(3))
print("3-dec share by M quantile:", pd.Series(c3).groupby(pd.qcut(M, 5, duplicates='drop')).mean().round(3).to_dict())
print("3-dec share by zone: min/median/max", pd.Series(c3).groupby(tr.zone_region_id.values).mean().describe()[["min", "50%", "max"]].round(3).to_dict())
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]
yH = d["elec_y"]; pc = d["elec_corrected"]; lr = np.log(yH / pc)
cH = c3[H]
print("H elec log-res mean/std 3-dec", lr[cH].mean().round(4), lr[cH].std().round(4), " 1-dec", lr[~cH].mean().round(4), lr[~cH].std().round(4))
g = d["gas_y"]; pg = d["gas_corrected"]; lg = np.log(g / pg)
print("H gas log-res std 3-dec", lg[cH].std().round(4), " 1-dec", lg[~cH].std().round(4))
