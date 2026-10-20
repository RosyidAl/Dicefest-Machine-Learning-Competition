"""Integrity audit 3: hard upper bound of gas (train gas < 1e6, max 999,765) and the implied EUI bound 1e6/A.

Honest by construction for the hard clip (no parameter fitted on H): clip the honest chain prediction on H at 1e6 (gas)
and 1e6/A (EUI) and score on H. Also reports the same counts on the submitted test predictions.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_clip.py
"""
import json

import numpy as np
import pandas as pd

SIG = dict(gas=151750.04, eui=40.65, elec=41725.37)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, A = z["H"], z["A"]
AH = A[H]
tr = pd.read_parquet("data/cache/train.parquet", columns=["target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm",
                                                          "target_macro_elec_kwh"])
out = {}
out["gas_y_matches_train"] = float(np.max(np.abs(z["gas_y"] - tr["target_macro_gas_kwh"].values[H])))
out["eui_y_matches_train"] = float(np.max(np.abs(z["eui_y"] - tr["target_eui_gas_kwh_per_sqm"].values[H])))
out["elec_y_matches_train"] = float(np.max(np.abs(z["elec_y"] - tr["target_macro_elec_kwh"].values[H])))


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


for cap in (1e6, 999764.8988254856):
    res = {}
    for m in ("gas", "eui"):
        y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
        hi = cap if m == "gas" else cap / AH
        pc = np.minimum(p, hi)
        res[m] = dict(n_above=int((p > hi).sum()), base=nr(y, p, m), clipped=nr(y, pc, m),
                      delta_total=(nr(y, pc, m) - nr(y, p, m)) / 5)
    out[f"H_cap_{cap:.0f}"] = res

# test side (submitted file)
te = pd.read_parquet("data/cache/test.parquet", columns=["bld_dim_06"])
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
At = te["bld_dim_06"].values
g, e = sub["target_macro_gas_kwh"].values, sub["target_eui_gas_kwh_per_sqm"].values
out["test_gas_above_1e6"] = int((g > 1e6).sum())
out["test_gas_above_1e6_values"] = sorted(g[g > 1e6].round(0).tolist())
out["test_eui_above_1e6_over_A"] = int((e > 1e6 / At).sum())
out["test_eui_excess_over_bound"] = sorted((e - 1e6 / At)[e > 1e6 / At].round(2).tolist())
# implied gas from EUI column vs gas column for the rows above the bound
out["test_gas_rows_eui_times_A"] = sorted((e * At)[g > 1e6].round(0).tolist())
# H: truncation bias near the bound (mean residual of rows with high predictions)
y, p = z["gas_y"], z["gas_corrected"]
for lo in (8e5, 9e5, 9.5e5):
    msk = p > lo
    out[f"H_gas_pred_gt_{lo:.0f}"] = dict(n=int(msk.sum()), mean_resid=float(np.mean(y[msk] - p[msk])) if msk.any() else 0)
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_clip.json", "w"), indent=1)
