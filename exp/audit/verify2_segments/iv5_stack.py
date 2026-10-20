"""Independent re-verification (round 'iv'): is the claimed calibration already inside the current best submissions, and
what would stacking it a second time cost? Light inputs only (submission CSVs, chain_H.npz).
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_segments/iv5_stack.py
"""
import hashlib

import numpy as np
import pandas as pd
from scipy.special import erfcx

GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}


def fnew(g, k=0.08, cap=1e6):
    out = g.copy()
    m = g > 6e5
    s = k * g[m]
    a = (cap - g[m]) / s
    out[m] = g[m] - s * np.sqrt(2.0 / np.pi) / erfcx(-a / np.sqrt(2.0))
    return out


te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06"])
At = te["bld_dim_06"].values
base = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
for f in ("submissions/sub_v19vce.csv", "exp/audit/segments/sub_v19vce_cap_k0.08.csv", "submissions/sub_v19ncapp.csv",
          "submissions/sub_v19nca2pp.csv", "submissions/BEST_CURRENT.csv"):
    s = pd.read_csv(f, float_precision="round_trip")
    g = s[GAS].values
    gE = s[EUI].values * At
    sha = hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
    again = fnew(g)
    print(f"{f:48s} sha {sha}  gas max {g.max():9.0f}  >0.95e6 {(g > 0.95e6).sum():3d}  >1e6 {(g > 1e6).sum()}  "
          f"eui*A max {gE.max():9.0f}  | k=0.08 applied on top: rows moved > 0.1 sigma {(np.abs(again - g) > 0.1 * SIG['gas']).sum():3d}, "
          f"rms move/sigma {np.sqrt(np.mean((again - g) ** 2)) / SIG['gas']:.5f}")
# holdout proxy for stacking: calibration once vs twice on H corrected
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
for met, sc in (("gas", 1.0), ("eui", z["A"][z["H"]])):
    y, c = z[f"{met}_y"], z[f"{met}_corrected"]
    n = len(y)

    def d(p):
        return (np.sqrt(np.mean((y - p) ** 2)) - np.sqrt(np.mean((y - c) ** 2))) / SIG[met] / 5

    once = fnew(c * sc) / sc
    twice = fnew(fnew(c * sc)) / sc
    print(f"H corrected {met}: k 0.08 once dTotal {d(once):+.8f}  twice (stacked) {d(twice):+.8f}")
