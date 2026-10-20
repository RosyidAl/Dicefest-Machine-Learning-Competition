"""Exact collisions of ratio quantities (gas/M, gas/E, EUI, gas/b03, elec/M, elec/E) between different train rows,
excluding the K0 exact duplicates and plain gas twins. A cluster of exact ratio collisions would mean a shared per-meter source."""
import numpy as np, pandas as pd
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=K0 + ["bld_dim_03", "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"])
k = pd.util.hash_pandas_object(tr[K0], index=False); dup = k.map(k.value_counts()).values > 1
g = tr.target_macro_gas_kwh.values; e = tr.target_macro_elec_kwh.values
gt = pd.Series(g).map(pd.Series(g).value_counts()).values > 1
Q = {"gas/M": g / tr.bld_dim_12.values, "gas/E": g / tr.bld_dim_01.values, "eui": tr.target_eui_gas_kwh_per_sqm.values, "gas/b03": g / tr.bld_dim_03.values,
     "elec/M": e / tr.bld_dim_12.values, "elec/E": e / tr.bld_dim_01.values}
for n, q in Q.items():
    s = pd.Series(np.round(q, 9))
    rep = s.map(s.value_counts()).values > 1
    print(f"{n}: rows in exact-collision groups {rep.sum()}, excluding K0 dups {int((rep & ~dup).sum())}, excluding K0 and gas twins {int((rep & ~dup & ~gt).sum())}")
