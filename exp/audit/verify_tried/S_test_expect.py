"""Expected test-set change of the small-meter fix S as built in the audit candidate (P-chain factors, s = 0.5), with
the bias allowed to differ between seen and unseen subzones (test has far fewer unseen small-M rows than H).

expected delta SSE on test = sum over cells (group x seen/unseen) of sum_{test rows} c^2 [(g - b)^2 - (1 - b)^2],
g = 1 + 0.5 (f_P(group) - 1), b = least-squares bias of the cell estimated on
  (i) P teacher level (seen proxy = subzone has another P row), independent of H labels
  (ii) H corrected level (seen = subzone has a P row)
  (iii) one bias per group (no seen/unseen split) from P, as the auditor assumed
c = submitted test prediction (sub_v19vce). Base MSE = H corrected MSE (approximation of the test base).
usage: python exp/audit/verify_tried/S_test_expect.py
"""
import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
COL = {"gas": "target_macro_gas_kwh", "elec": "target_macro_elec_kwh", "eui": "target_eui_gas_kwh_per_sqm"}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M, subn = z["P"], z["H"], z["M"], z["subn"]
MP, MH = M[P], M[H]
GROUPS = (("<=3", lambda v: v <= 3), ("4", lambda v: v == 4))
unseenH = ~np.isin(subn[H], np.unique(subn[P]))
u, cnt = np.unique(subn[P], return_counts=True)
singleP = np.isin(subn[P], u[cnt == 1])
te = pd.read_parquet("cache/test.parquet", columns=["id", "subzone_id", "bld_dim_12"])
subnt = te["subzone_id"].str[4:].astype(int).values
Mt = te["bld_dim_12"].values.astype(float)
unseenT = ~np.isin(subnt, np.unique(subn))
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
nT = len(te)
tot = {"P cells": 0.0, "H cells": 0.0, "P group": 0.0}
for m in SIG:
    pP, yP = z[f"{m}_pP"], z[f"{m}_rP"] + z[f"{m}_pP"]
    y, c = z[f"{m}_y"], z[f"{m}_corrected"]
    mse = np.mean((y - c) ** 2)
    ct = sub[COL[m]].values
    out = {}
    for name in tot:
        dsse = 0.0
        for k, g in GROUPS:
            fP = yP[g(MP)] @ pP[g(MP)] / (pP[g(MP)] @ pP[g(MP)])
            gg = 1 + 0.5 * (fP - 1)
            for seen in (True, False):
                st = g(Mt) & (unseenT != seen)
                if name == "P cells":
                    s = g(MP) & (singleP != seen)
                    b = yP[s] @ pP[s] / (pP[s] @ pP[s])
                elif name == "H cells":
                    s = g(MH) & (unseenH != seen)
                    b = y[s] @ c[s] / (c[s] @ c[s])
                else:
                    b = fP
                dsse += np.sum(ct[st] ** 2) * ((gg - b) ** 2 - (1 - b) ** 2)
        out[name] = (np.sqrt(mse + dsse / nT) - np.sqrt(mse)) / SIG[m] / 5
        tot[name] += out[name]
    print(f"{m}: " + ", ".join(f"{k} {v:+.8f}" for k, v in out.items()), flush=True)
print("expected test TOTAL: " + ", ".join(f"{k} {v:+.8f}" for k, v in tot.items()), flush=True)
