"""Numeric precision structure of targets: are gas, gas/M, EUI, elec, elec/M 'round' numbers? (hints at how values were generated)"""
import numpy as np, pandas as pd
T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_12", "bld_dim_01", "bld_dim_06", "bld_dim_03"] + T)
g, e, u = (tr[c].values for c in T); M = tr.bld_dim_12.values; A = tr.bld_dim_06.values
def roundness(x, name):
    out = []
    for d in [0, 1, 2, 3, 4, 6]:
        out.append(f"d{d}:{np.mean(np.abs(x * 10**d - np.round(x * 10**d)) < 1e-3):.3f}")
    print(name, " ".join(out))
roundness(g, "gas"); roundness(g / M, "gas/M"); roundness(u, "eui"); roundness(e, "elec"); roundness(e / M, "elec/M")
roundness(A, "A"); roundness(g / tr.bld_dim_01.values, "gas/E")
print("gas samples", g[:5], "gas/M", (g/M)[:5], "eui", u[:5])
# elec twins: how many identical elec values (0.1 grid) expected by chance?
vc = pd.Series(e).value_counts(); print("elec distinct", len(vc))
