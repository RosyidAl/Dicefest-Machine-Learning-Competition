"""verify2_segments, part A: check the claimed blunder on the actual files, independently of the segments lens.

Claim under test: 7 test gas predictions in submissions/sub_v19vce.csv exceed 1e6 (range 1,004,360..1,061,410),
while the gas target is truncated below 1e6, so they are certainly too high; 8 test EUI predictions have EUI*A > 1e6.

Reads the raw CSVs (column subsets only, not the parquet cache) so the truncation evidence does not depend on the
cache. Prints a short report; run with stdout redirected to v2s_file_check.log.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_segments/v2s_file_check.py
"""
import os

import numpy as np
import pandas as pd

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
CAP = 1e6
RT = dict(float_precision="round_trip")

tr = pd.read_csv("data/raw/train.csv", usecols=["id", "bld_dim_06", GAS, EUI], **RT)
te = pd.read_csv("data/raw/test.csv", usecols=["id", "bld_dim_06"], **RT)
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
print(f"train rows {len(tr)}  test rows {len(te)}  sample rows {len(ss)}")

# ---------- 1. truncation evidence on the train labels ----------
g = tr[GAS].values.astype(float)
A = tr["bld_dim_06"].values.astype(float)
e = tr[EUI].values.astype(float)
print("\n[train] gas max %.4f  rows >= 1e6: %d  rows >= 999,000: %d" % (g.max(), (g >= CAP).sum(), (g >= 999e3).sum()))
rel = np.abs(e * A - g) / g
print("[train] identity eui*A == gas: max rel err %.3e  max eui*A %.4f" % (rel.max(), (e * A).max()))
edges = np.arange(700e3, 1000e3 + 1, 25e3)
cnt, _ = np.histogram(g, bins=edges)
print("[train] counts per 25k bin from 700k to 1e6:", " ".join(f"{int(c)}" for c in cnt))
# log-linear tail fit on the 700k..1e6 bins, extrapolated to [1e6, 1.1e6): expected count if there were no cap
mid = 0.5 * (edges[1:] + edges[:-1])
b, a = np.polyfit(mid, np.log(np.maximum(cnt, 1)), 1)
ext = np.exp(a + b * np.arange(1012.5e3, 1100e3, 25e3)).sum()
print(f"[train] log-linear tail extrapolation: expected rows in [1e6,1.1e6) without a cap ~ {ext:.0f} (observed 0)")
top = np.sort(g)[::-1][:5]
print("[train] top 5 gas:", " ".join(f"{v:.1f}" for v in top))

# ---------- 2. the submission ----------
results = {}
files = ["submissions/sub_v19vce.csv", "submissions/BEST_CURRENT.csv", "submissions/sub_v19vc.csv",
         "submissions/sub_v19a3.csv", "exp/audit/segments/sub_v19vce_cap_k0.08.csv"]
At = te["bld_dim_06"].values.astype(float)
for f in files:
    if not os.path.exists(f):
        print(f"\n[{f}] missing")
        continue
    sub = pd.read_csv(f, usecols=["id", GAS, EUI], **RT)
    assert len(sub) == len(te) and (sub["id"].values == ss["id"].values).all() and (sub["id"].values == te["id"].values).all()
    pg, pe = sub[GAS].values, sub[EUI].values
    over = np.where(pg > CAP)[0]
    over_max = np.where(pg > g.max())[0]
    over_e = np.where(pe * At > CAP)[0]
    print(f"\n[{f}] gas > 1e6: {len(over)} rows; gas > train max: {len(over_max)} rows; eui*A > 1e6: {len(over_e)} rows;"
          f" gas max {pg.max():.1f}; max eui*A {np.max(pe * At):.1f}")
    if f.endswith("sub_v19vce.csv"):
        o = over[np.argsort(-pg[over])]
        for i in o:
            print(f"   id {sub['id'].values[i]:>10s}  gas {pg[i]:12.1f}  eui {pe[i]:9.3f}  A {At[i]:8.1f}  eui*A {pe[i] * At[i]:12.1f}"
                  f"  excess/sigma {(pg[i] - CAP) / 151750.04:.4f}")
        oe = over_e[np.argsort(-(pe * At)[over_e])]
        print("   eui rows with eui*A > 1e6 (id, eui, cap 1e6/A, excess/sigma):")
        for i in oe:
            print(f"   id {sub['id'].values[i]:>10s}  eui {pe[i]:9.3f}  cap {CAP / At[i]:9.3f}  excess/sigma {(pe[i] - CAP / At[i]) / 40.65:.4f}"
                  f"  gas pred {pg[i]:12.1f}")
        # are any of these the 77 duplicate-copied rows? copied rows carry a train value, which is < 1e6 by section 1
        print("   any over-cap gas value equal to a train gas value:", bool(np.isin(pg[over], g).any()))
        np.save("exp/audit/verify2_segments/test_over_idx.npy", over)
        np.save("exp/audit/verify2_segments/test_over_eui_idx.npy", over_e)
