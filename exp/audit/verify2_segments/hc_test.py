"""verify2 (segments lens): expected effect of the hard clip on the 7 gas / 8 EUI test rows of sub_v19vce.

Per row the change of squared error is (c - p)(c + p - 2y) for p > c > y: strictly negative whenever the truth y < c,
so the sign is fixed; only the size depends on y. Bounds: y = c (smallest gain) and y = E[y | p > 1e6] estimated on
honest held-out rows (H corrected; P teacher as a second estimate) from hc_chain.json.
dRMSE is converted with the H-chain RMSE as the base (gas 0.24514, EUI 0.64859 NRMSE), over all 183,987 test rows.
"""
import json

import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65}
BASE = {"gas": 0.24514, "eui": 0.64859}
CAP = 1e6
s = pd.read_csv("submissions/sub_v19vce.csv", usecols=["id", "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06"])
assert np.array_equal(s["id"].to_numpy(), te["id"].to_numpy())
A = te["bld_dim_06"].to_numpy()
n = len(s)
ch = json.load(open("exp/audit/verify2_segments/hc_chain.json"))
g = s["target_macro_gas_kwh"].to_numpy()
e = s["target_eui_gas_kwh_per_sqm"].to_numpy()
ig, ie = np.where(g > CAP)[0], np.where(e * A > CAP)[0]
print(f"test rows: gas > 1e6 {len(ig)}, eui*A > 1e6 {len(ie)}")


def d_total(dsse, m):
    mse0 = (BASE[m] * SIG[m]) ** 2
    return (np.sqrt(mse0 + dsse / n) - np.sqrt(mse0)) / SIG[m] / 5


for tag, yg in [("y = 1e6 (minimum gain)", CAP), ("y = E[y|p>1e6] H corrected", ch["cond H corrected"]["mean_y"]),
                ("y = E[y|p>1e6] P teacher", ch["cond P teacher"]["mean_y"]), ("y = 900k", 9e5)]:
    pg = g[ig]
    dg = np.sum((CAP - pg) * (CAP + pg - 2 * yg))
    ce, pe = CAP / A[ie], e[ie]
    ye = np.minimum(yg, CAP) / A[ie]
    de = np.sum((ce - pe) * (ce + pe - 2 * ye))
    tg, tE = d_total(dg, "gas"), d_total(de, "eui")
    print(f"  {tag:<30s} gas dTotal {tg:+.8f}  eui dTotal {tE:+.8f}  sum {tg + tE:+.8f}")
