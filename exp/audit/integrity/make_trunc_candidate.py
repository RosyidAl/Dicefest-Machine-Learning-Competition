"""Integrity audit 11: candidate = sub_v19vce + truncation adjustment of gas and EUI near gas = 1e6 kWh.

All parameters were fitted on P-side teacher residuals only (check_trunc2.py, normal model):
  s = p_gas * sqrt(a0 + b0/M), adjustment = -c * s * phi(a)/Phi(a), a = (1e6 - p_gas)/s
  gas: p += c_gas * adj(p);  EUI: p = (p*A + c_eui * adj(p*A)) / A
The 77 rows copied from train duplicates are left untouched (their values are exact).
Also checks that the unseen-subzone adoption equals clip(adopt_v2.npy, 0, 100).
Writes exp/audit/integrity/sub_v19vce_trunc.csv (NOT a submission; nothing is submitted).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/make_trunc_candidate.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

p2 = json.load(open("exp/audit/integrity/check_trunc2.json"))
a0, b0 = p2["var_normal"]
c_gas, c_eui = p2["normal"]["gas"]["c_P"], p2["normal"]["eui"]["c_P"]
T = 1e6
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
GAS, EUI, AD = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm", "target_electric_heating_adoption"
tr = pd.read_parquet("data/cache/train.parquet", columns=KEY + [GAS])
te = pd.read_parquet("data/cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
out = {}
# adoption check on unseen subzones
seen = te["subzone_id"].isin(tr["subzone_id"]).values
ad = np.clip(np.load("exp/pred/adopt_v2.npy"), 0, 100)
out["unseen_adoption_equals_adopt_v2"] = int((sub[AD].values[~seen] == ad[~seen]).sum())
out["unseen_rows"] = int((~seen).sum())
# duplicate rows (exact train targets) stay untouched
ref = tr.groupby(KEY, sort=False)[GAS].first().reset_index()
hit = ~te[KEY].merge(ref, on=KEY, how="left")[GAS].isna().values
out["dup_rows_untouched"] = int(hit.sum())
A, M = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)


def adj(pg):
    s = np.maximum(np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / M, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


new = sub.copy()
g = sub[GAS].values
e = sub[EUI].values
g2 = g + c_gas * adj(g)
e2 = (e * A + c_eui * adj(e * A)) / A
new[GAS] = np.where(hit, g, np.clip(g2, 0, None))
new[EUI] = np.where(hit, e, np.clip(e2, 0, None))
dg = new[GAS].values - g
de = new[EUI].values - e
out["params"] = dict(a0=a0, b0=b0, c_gas=c_gas, c_eui=c_eui)
out["gas_rows_changed_gt_1kwh"] = int((np.abs(dg) > 1).sum())
out["gas_rows_changed_gt_1pct"] = int((np.abs(dg) > 0.01 * g).sum())
out["gas_change_min_max"] = [float(dg.min()), float(dg.max())]
out["gas_rms_change_over_sigma"] = float(np.sqrt(np.mean(dg ** 2)) / 151750.04)
out["eui_rows_changed_gt_0.01"] = int((np.abs(de) > 0.01).sum())
out["eui_rms_change_over_sigma"] = float(np.sqrt(np.mean(de ** 2)) / 40.65)
out["gas_max_after"] = float(new[GAS].max())
out["eui_times_A_max_after"] = float((new[EUI].values * A).max())
path = "exp/audit/integrity/sub_v19vce_trunc.csv"
new.to_csv(path, index=False)
out["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
out["other_columns_identical"] = {c: bool((new[c].values == sub[c].values).all())
                                  for c in ["target_macro_elec_kwh", "target_clean_building_ratio", AD]}
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/make_trunc_candidate.json", "w"), indent=1)
