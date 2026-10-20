"""Integrity audit 5: gas-only twins (rows sharing an exact gas value but not elec). Structure check.

group sizes; do twins share M (bld_dim_12), A (bld_dim_06), E (bld_dim_01), subzone, zone, climate cell, gas per meter?
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_twins.py
"""
import json

import numpy as np
import pandas as pd

cols = ["id", "subzone_id", "zone_region_id", "bld_dim_12", "bld_dim_06", "bld_dim_01", "bld_dim_02", "bld_dim_03",
        "env_climate_01", "env_climate_03", "env_climate_05", "socio_demog_005", "target_macro_gas_kwh",
        "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
tr = pd.read_parquet("data/cache/train.parquet", columns=cols)
g = tr["target_macro_gas_kwh"]
vc = g.value_counts()
tw = tr[g.map(vc).values > 1].copy()
out = {"twin_rows": len(tw), "twin_groups": int((vc > 1).sum()),
       "group_size_dist": {int(k): int(v) for k, v in vc[vc > 1].value_counts().sort_index().items()}}
out["top_values"] = {repr(float(k)): int(v) for k, v in vc.head(8).items()}
grp = tw.groupby("target_macro_gas_kwh")
same = {}
for c in ["subzone_id", "zone_region_id", "bld_dim_12", "bld_dim_06", "bld_dim_01", "bld_dim_02", "bld_dim_03",
          "env_climate_01", "target_macro_elec_kwh"]:
    same[c] = float((grp[c].nunique() == 1).mean())
out["share_groups_with_same"] = same
# decimals of the gas value: how 'round' are twin values vs non-twin
def ndec(x):
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s else 0
out["twin_value_decimals_median"] = float(np.median([ndec(x) for x in vc[vc > 1].index[:2000]]))
out["nontwin_value_decimals_median"] = float(np.median([ndec(x) for x in vc[vc == 1].index[:2000]]))
# gas per meter for twins: identical?
tw["gpm"] = tw["target_macro_gas_kwh"] / tw["bld_dim_12"]
out["twin_rows_M_dist"] = tw["bld_dim_12"].describe().round(2).to_dict()
out["all_rows_M_dist"] = tr["bld_dim_12"].describe().round(2).to_dict()
# example groups
ex = []
for v, d in list(grp)[:6]:
    ex.append(d[["subzone_id", "bld_dim_12", "bld_dim_06", "bld_dim_01", "target_macro_elec_kwh"]].astype(str).values.tolist())
out["examples"] = ex
# is the shared gas value equal to gas of some row times a ratio? check gas twins vs integer gas
out["twin_values_integer_share"] = float(np.mean(np.isclose(vc[vc > 1].index % 1, 0)))
out["all_values_integer_share"] = float(np.mean(np.isclose(g.values % 1, 0)))
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_twins.json", "w"), indent=1)
