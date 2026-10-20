"""Rebuild the S1-only candidate with round-trip float parsing (so only the 358 shifted rows per metric differ from
submissions/sub_v19vce.csv), and count how many other rows the prior candidate sub_v19vce_pp.csv changes."""
import hashlib
import json

import numpy as np
import pandas as pd

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
COL = {"gas": GAS, "eui": EUI, "elec": EL}
D = json.load(open("exp/audit/verify2_postproc/v02_fulloof.json"))["deltas_all_rows"]
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_12"])
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
Mt = te["bld_dim_12"].values.astype(float)
G = {"G23": (Mt >= 2) & (Mt <= 3), "G4": Mt == 4}  # no duplicate rows in these groups (v03)
new = sub.copy()
for m, c in COL.items():
    p = sub[c].values.astype(float).copy()
    for g, msk in G.items():
        p[msk] = p[msk] * (1 + D[m][g])
    new[c] = np.clip(p, 0, None)
fn = "exp/audit/verify2_postproc/sub_v19vce_s1only.csv"
new.to_csv(fn, index=False)
chk = pd.read_csv(fn, float_precision="round_trip")
res = {"changed_rows": {c: int((chk[c].values != sub[c].values).sum()) for c in sub.columns[1:]},
       "sha256": hashlib.sha256(open(fn, "rb").read()).hexdigest()}
pp = pd.read_csv("exp/audit/postproc/sub_v19vce_pp.csv", float_precision="round_trip")
s1 = G["G23"] | G["G4"]
res["prior_pp_nonS1_rows_changed_rel_gt_1e-12"] = {
    m: int((np.abs(pp[c].values[~s1] / sub[c].values[~s1] - 1) > 1e-12).sum()) for m, c in COL.items()}
res["prior_pp_nonS1_rows_changed_rel_gt_1e-6"] = {
    m: int((np.abs(pp[c].values[~s1] / sub[c].values[~s1] - 1) > 1e-6).sum()) for m, c in COL.items()}
print(json.dumps(res, indent=1))
json.dump(res, open("exp/audit/verify2_postproc/v04_s1only_exact.json", "w"), indent=1)
