"""Truncation evidence in the train targets and prediction bias near the candidate bounds.

Candidate data filters (from the train target distribution only): gas < 1e6, elec / A < 150, gas / A > 5, elec / A > 5.
For each: density of train rows near the bound (sharp edge = truncation), then mean residual of the P-side teacher
(OOF blend) and of the honest H corrected prediction in bins of the predicted distance to the bound.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_trunc.py > exp/audit/segments/seg_trunc.log
"""
import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
COL = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
ytr = pd.read_parquet("cache/train.parquet", columns=list(COL.values()))
gas, elec, eui = (ytr[COL[k]].values for k in ("gas", "elec", "eui"))

print("== edge densities (train, all rows) ==")
for name, v, edges in [("gas", gas, [9.0e5, 9.5e5, 9.8e5, 9.9e5, 1.0e6, 1.01e6]),
                       ("elec/A", elec / A, [130, 140, 145, 148, 149, 149.5, 150, 151]),
                       ("gas/A", gas / A, [4.5, 5, 5.5, 6, 7, 8, 10]),
                       ("elec/A low", elec / A, [4.5, 5, 5.5, 6, 7, 8, 10]),
                       ("elec", elec, [3.0e5, 3.2e5, 3.4e5, 3.6e5, 3.8e5, 4.0e5])]:
    h = np.histogram(v, bins=edges)[0]
    print(f"  {name:10s} bins {edges}: counts {h.tolist()}")
for name, v in [("gas", gas), ("elec/A", elec / A), ("gas/A", gas / A), ("elec", elec)]:
    s = np.sort(v)
    print(f"  {name:8s} lowest 5 {np.round(s[:5], 4).tolist()}  highest 5 {np.round(s[-5:], 4).tolist()}")


def table(label, y, p, d, bins):
    """mean residual (y - p) by bins of d"""
    g = np.digitize(d, bins)
    df = pd.DataFrame({"g": g, "e": y - p})
    s = df.groupby("g")["e"].agg(["size", "mean", "std"])
    s["t"] = s["mean"] / (s["std"] / np.sqrt(s["size"]))
    print(f"  {label}")
    for gi, r in s.iterrows():
        lo = bins[gi - 1] if gi > 0 else -np.inf
        hi = bins[gi] if gi < len(bins) else np.inf
        print(f"     [{lo:>8.4g},{hi:>8.4g})  n {int(r['size']):6d}  mean {r['mean']:+10.4g}  t {r['t']:+6.2f}")


print("\n== gas near the 1e6 cap: residual by predicted p / 1e6 ==")
b = [0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02]
table("P teacher (OOF blend), gas", gas[P], z["gas_pP"], z["gas_pP"] / 1e6, b)
table("H corrected, gas", z["gas_y"], z["gas_corrected"], z["gas_corrected"] / 1e6, b)
table("H corrected eui * A (eui column scaled), gas units", z["eui_y"] * A[H], z["eui_corrected"] * A[H],
      z["eui_corrected"] * A[H] / 1e6, b)

print("\n== elec near elec/A = 150: residual by predicted (p / A) ==")
b = [80, 100, 110, 120, 130, 140, 145, 150, 160]
table("P teacher, elec", elec[P], z["elec_pP"], z["elec_pP"] / A[P], b)
table("H corrected, elec", z["elec_y"], z["elec_corrected"], z["elec_corrected"] / A[H], b)

print("\n== gas/eui near gas/A = 5 and elec near elec/A = 5 (floor) ==")
b = [5, 7, 10, 15, 20, 30]
table("P teacher, eui (p = gas EUI)", eui[P], z["eui_pP"], z["eui_pP"], b)
table("H corrected, eui", z["eui_y"], z["eui_corrected"], z["eui_corrected"], b)
table("H corrected, elec by p/A", z["elec_y"], z["elec_corrected"], z["elec_corrected"] / A[H], b)
