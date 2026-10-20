"""verify2_integrity / step 6: is gas the only target with a hard bound? Density of the top / bottom 0.2% of each
target-derived quantity in 10 equal-width bins (a hard filter shows a flat density ending abruptly at the extreme,
a natural tail decays to a few rows).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/v6_other_bounds.py
"""
import json

import numpy as np
import pandas as pd

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EL, EUI, "bld_dim_06", "bld_dim_12", "bld_dim_01"])
M, A, E = tr["bld_dim_12"].values.astype(float), tr["bld_dim_06"].values, tr["bld_dim_01"].values.astype(float)
q = {"gas": tr[GAS].values, "elec": tr[EL].values, "eui": tr[EUI].values, "gas_per_meter": tr[GAS].values / M,
     "elec_per_meter": tr[EL].values / M, "elec_per_A": tr[EL].values / A, "gas_per_epc": tr[GAS].values / E,
     "elec_per_epc": tr[EL].values / E}
res = {}
for name, v in q.items():
    hi = np.quantile(v, 0.998)
    lo = np.quantile(v, 0.002)
    res[name] = dict(max=float(v.max()), min=float(v.min()),
                     top_bins=np.histogram(v[v >= hi], 10)[0].tolist(),
                     bottom_bins=np.histogram(v[v <= lo], 10)[0].tolist())
    print(f"{name:>15} max {v.max():12.2f} top {res[name]['top_bins']}  min {v.min():10.2f} bottom {res[name]['bottom_bins']}")
json.dump(res, open("exp/audit/verify2_integrity/v6_other_bounds.json", "w"), indent=1)
