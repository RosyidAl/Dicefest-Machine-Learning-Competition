"""AUDIT CANDIDATE (not pipeline-reproduced; for a public-LB probe only): sub_v19vce + the two rare-segment fixes,
parameters fitted on P of the holdout chain (exp/audit/tried/combo_params_P.json):
  S: rows with bld_dim_12 <= 4: gas / EUI / elec multiplied by 1 + 0.5 * (f - 1), f per metric and group {<=3, 4};
  G: gas above 9e5 shrunk: g' = min(g + beta * (g - 9e5), 1e6); EUI rows whose implied gas e*A > 9e5 get the same fix.
The 77 duplicate rows (exact train copies, src/dup_fix.py key) are left untouched. Output: exp/audit/tried/sub_v19vce_GS_audit.csv"""
import hashlib
import json

import numpy as np
import pandas as pd

T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm", "target_clean_building_ratio",
     "target_electric_heating_adoption"]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
prm = json.load(open("exp/audit/tried/combo_params_P.json"))
beta, KNOT, CAP, fac = prm["beta"], prm["knot"], prm["cap"], prm["factors"]
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
tr = pd.read_parquet("cache/train.parquet", columns=KEY)
assert (sub["id"].values == te["id"].values).all()
dup = te[KEY].merge(tr.drop_duplicates(KEY).assign(_d=1), on=KEY, how="left")["_d"].notna().values
print("duplicate rows (untouched):", int(dup.sum()))
A, M = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
out = sub.copy()
col = {"gas": T[0], "elec": T[1], "eui": T[2]}
for m, c in col.items():
    v = out[c].values.copy()
    for k, sel in (("<=3", M <= 3), ("4", M == 4)):
        s = sel & ~dup
        v[s] = v[s] * (1 + 0.5 * (fac[m][k] - 1))
    if m == "gas":
        s = (v > KNOT) & ~dup
        v[s] = np.minimum(v[s] + beta * (v[s] - KNOT), CAP)
    elif m == "eui":
        g = v * A
        s = (g > KNOT) & ~dup
        v[s] = np.minimum(g[s] + beta * (g[s] - KNOT), CAP) / A[s]
    out[c] = np.clip(v, 0, None)
    ch = out[c].values != sub[c].values
    print(f"{m}: rows changed {int(ch.sum())}, rms change / sigma "
          f"{np.sqrt(np.mean((out[c].values - sub[c].values) ** 2)) / {'gas': 151750.04, 'eui': 40.65, 'elec': 41725.37}[m]:.6f}")
print("test rows with M<=3:", int((M <= 3).sum()), "M==4:", int((M == 4).sum()), "gas>9e5:", int((sub[T[0]].values > KNOT).sum()))
assert (out[T[3:]].values == sub[T[3:]].values).all()
path = "exp/audit/tried/sub_v19vce_GS_audit.csv"
out.to_csv(path, index=False)
print("wrote", path, hashlib.sha256(open(path, "rb").read()).hexdigest())
