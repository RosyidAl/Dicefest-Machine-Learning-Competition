"""Power of the train-only-filter check in v1 and a larger-sample truth-gap estimate.

 - Among train rows with true gas in [950k, 1e6) (the nearest analog of rows a train-only filter would have removed),
   what share has proxy prediction (equal-weight mean of the v19vce gas members, OOF) above 7e5 / 9e5?
   -> expected test excess in those bins under the alternative.
 - E[1e6 - y | prediction above the cap] from the full-train OOF proxy (25 rows) and from the H chain teacher/corrected.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/v2_alt_power.py \
          > exp/audit/verify_segments/v2_alt_power.log
"""
import numpy as np
import pandas as pd
import yaml

GAS = "target_macro_gas_kwh"
CAP = 1e6
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
A, H = z["A"], z["H"]
yg = pd.read_parquet("cache/train.parquet", columns=[GAS])[GAS].values
members = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
oof = np.mean([np.load(f"exp/oof/{r}.npy") for r in members], axis=0) * A
ntr, nte = 429303, 183987
top = (yg >= 9.5e5)
for thr in (7e5, 8e5, 9e5, 1e6):
    print(f"truth in [950k,1e6): {top.sum()} rows; share with OOF proxy > {thr:.0e}: {np.mean(oof[top] > thr):.3f}")
alt_rows = 343 * nte / ntr  # from v1 exponential-tail extrapolation
for thr, n_te_obs, n_tr_obs in [(7e5, 1652, 3845), (9e5, 97, 226)]:
    share = np.mean(oof[top] > thr)  # rows above the cap would score at least this high
    exp_null = n_tr_obs * nte / ntr
    exp_alt = exp_null + alt_rows * share
    print(f"proxy > {thr:.0e}: test observed {n_te_obs}, null {exp_null:.0f}, alternative >= {exp_alt:.0f} "
          f"(z of alt vs observed {(n_te_obs - exp_alt) / np.sqrt(exp_alt):+.1f})")
sel = oof > CAP
print(f"\nfull-train OOF proxy > 1e6: {sel.sum()} rows; gap 1e6 - y: mean {np.mean(CAP - yg[sel]):.0f} "
      f"median {np.median(CAP - yg[sel]):.0f}")
for name in ("gas_teacher", "gas_corrected"):
    p = z[name]
    for lo, hi in [(0.98e6, 1e6), (1e6, 1.02e6), (1.02e6, 1.1e6)]:
        s = (p > lo) & (p <= hi)
        if s.any():
            print(f"H {name:13s} pred in ({lo:.3g},{hi:.3g}]: n {s.sum():3d} mean gap {np.mean(CAP - z['gas_y'][s]):8.0f}"
                  f"  mean (pred - y) {np.mean(p[s] - z['gas_y'][s]):8.0f}")
