"""Reference values of the G / S parameters on the full-train OOF of the deployed v19 members with the GLOBAL NNLS
weights (summary.json). The pipeline should fit them on its VC-blend fit predictions pf; values there are expected
between these and the holdout-chain P values (exp/audit/tried/combo_params_P.json)."""
import json

import numpy as np
import pandas as pd

s = json.load(open("final/work/v19vce/summary.json"))
T = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
tr = pd.read_parquet("cache/train.parquet", columns=list(T.values()) + ["bld_dim_06", "bld_dim_12"])
A, M = tr.bld_dim_06.values, tr.bld_dim_12.values
out = {}
for m in T:
    p = sum(np.load(f"exp/oof/{k}.npy") * v for k, v in s["weights"][m].items() if v > 0)
    p = p * A if m == "gas" else p
    y = tr[T[m]].values
    out[m] = {g: round(float(y[sel] @ p[sel] / (p[sel] @ p[sel])), 4) for g, sel in (("<=3", M <= 3), ("4", M == 4))}
    if m == "gas":
        h = np.maximum(p - 9e5, 0)
        out["beta"] = round(float(h @ (y - p) / (h @ h)), 4)
print("full-OOF global-blend parameters:", out)
print("holdout-chain P parameters:", json.load(open("exp/audit/tried/combo_params_P.json")))
