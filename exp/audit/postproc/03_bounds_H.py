"""Audit postproc 03: hard truncation bounds of the train targets applied to the honest holdout-chain predictions
(exp/audit/chain_H.npz, m_corrected = analog of sub_v19vce), and counts of violations in sub_v19vce.csv.

Bounds found in 02_tails (hard cuts, zero rows beyond): gas < 1e6; gas/A (= EUI) >= 5; 5 <= elec/A < 150.
On H the bound values are taken from P rows only (max/min of P targets), so nothing is fitted on H.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/03_bounds_H.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AP, AH = A[P], A[H]
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in SIG}  # metric scale (gas in kWh)
# P-derived bounds
b = dict(gas_max=yP["gas"].max(), eui_min=yP["eui"].min(), elecA_min=(yP["elec"] / AP).min(),
         elecA_max=(yP["elec"] / AP).max(), eui_max=yP["eui"].max())
print("P-derived bounds", {k: round(float(v), 4) for k, v in b.items()})
assert np.allclose(yP["gas"], yP["eui"] * AP, rtol=1e-6)


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


rules = {
    "gas": {"cap gas_max": (None, b["gas_max"]), "floor eui_min*A": (b["eui_min"] * AH, None),
            "cap eui_max*A": (None, b["eui_max"] * AH),
            "both": (b["eui_min"] * AH, b["gas_max"])},
    "eui": {"cap gas_max/A": (None, b["gas_max"] / AH), "floor eui_min": (b["eui_min"], None),
            "cap eui_max": (None, b["eui_max"]),
            "both": (b["eui_min"], np.minimum(b["gas_max"] / AH, b["eui_max"]))},
    "elec": {"cap 150*A": (None, b["elecA_max"] * AH), "floor 5*A": (b["elecA_min"] * AH, None),
             "both": (b["elecA_min"] * AH, b["elecA_max"] * AH)},
}
tot = 0.0
for m in ("gas", "eui", "elec"):
    y = z[f"{m}_y"]
    for stage in ("teacher", "corrected"):
        base = np.clip(z[f"{m}_{stage}"], 0, None)
        s0 = nr(y, base, m)
        print(f"{m} {stage}: base {s0:.6f}  (clip0 changed {int((z[f'{m}_{stage}'] < 0).sum())} rows)")
        for name, (lo, hi) in rules[m].items():
            p = np.clip(base, lo, hi)
            n = int((p != base).sum())
            d = nr(y, p, m) - s0
            print(f"   {name:18s} rows {n:5d}  dNRMSE {d:+.7f}  total {d / 5:+.8f}")
            if stage == "corrected" and name == "both":
                tot += d / 5
print(f"TOTAL (corrected, 'both' rules) {tot:+.8f}")

# test submission: violation counts against the full-train bounds
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06"])
sub = pd.read_csv("submissions/sub_v19vce.csv")
At = te["bld_dim_06"].values
Atr = tr["bld_dim_06"].values
B = dict(gas_max=tr[GAS].max(), eui_min=tr[EUI].min(), eui_max=tr[EUI].max(), elecA_min=(tr[EL] / Atr).min(),
         elecA_max=(tr[EL] / Atr).max())
print("full-train bounds", {k: round(float(v), 4) for k, v in B.items()})
g, e, el = sub[GAS].values, sub[EUI].values, sub[EL].values
print("test gas > gas_max", int((g > B["gas_max"]).sum()), "sum excess/sigma", float(np.clip(g - B["gas_max"], 0, None).sum() / SIG["gas"]))
print("test gas < eui_min*A", int((g < B["eui_min"] * At).sum()))
print("test eui > gas_max/A", int((e > B["gas_max"] / At).sum()), "excess", np.round(np.sort((e - B["gas_max"] / At)[e > B["gas_max"] / At]), 2))
print("test eui < eui_min", int((e < B["eui_min"]).sum()))
print("test eui > eui_max", int((e > B["eui_max"]).sum()))
print("test elec > elecA_max*A", int((el > B["elecA_max"] * At).sum()), "excess", np.round(np.sort((el - B["elecA_max"] * At)[el > B["elecA_max"] * At]), 0))
print("test elec < elecA_min*A", int((el < B["elecA_min"] * At).sum()), "deficit", np.round(np.sort((B["elecA_min"] * At - el)[el < B["elecA_min"] * At]), 0))
