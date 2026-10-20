"""Extra checks for the hard-clip claim (read-only; writes exp/audit/verify_integrity/clipR3_extra.json).

 1. Power of the "test is truncated at 1e6 too" check: how often rows with y near the bound get
    high predictions on H, and how many extra high test predictions an untruncated test would give.
 2. Whether later submissions (sub_v19ncapp, sub_v19nca2pp, BEST_CURRENT) still exceed the bound.
"""
import os
import json

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd

OUT = "exp/audit/verify_integrity/clipR3_extra.json"
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
C = 1e6
res = {}
d = np.load("exp/audit/chain_H.npz", allow_pickle=False)
y, p = d["gas_y"], d["gas_corrected"]
NH = len(y)
NT = 183987
ntr = 429303

# 1a. P(p > t | y in band) on H
band = {}
for lo, hi in ((900_000, 950_000), (950_000, 1_000_000)):
    sel = (y >= lo) & (y < hi)
    band[f"{lo}_{hi}"] = {"n": int(sel.sum()),
                          **{f"frac_p_gt_{t}": float((p[sel] > t).mean()) for t in (850_000, 900_000, 950_000, 1_000_000)}}
res["H_pred_given_y_band"] = band

# 1b. extra high predictions expected in test if test were NOT truncated.
# Untruncated tail: 387 train rows >= 1e6 by log-linear extrapolation (clipR3_verify.json), i.e. 387*NT/ntr in test.
v = json.load(open("exp/audit/verify_integrity/clipR3_verify.json"))
exp_train_ge = v["B_truncation"]["expected_train_rows_ge_1e6_if_tail_continued"]
extra_rows = exp_train_ge * NT / ntr
tails = v["D_test"]["pred_tail_counts"]
chk = {}
for t in (900_000, 950_000, 1_000_000):
    # conservative: rows with y >= 1e6 get p > t at least as often as rows with y in [950k, 1e6)
    f = band["950000_1000000"][f"frac_p_gt_{t}"]
    e_tr = tails[str(t)]["test_expected_from_H_rate"]
    obs = tails[str(t)]["test"]
    e_un = e_tr + extra_rows * f
    chk[str(t)] = {"observed_test": obs, "expected_if_truncated": e_tr, "expected_if_untruncated_min": e_un,
                   "z_obs_vs_untruncated": (obs - e_un) / np.sqrt(e_un),
                   "z_obs_vs_truncated": (obs - e_tr) / np.sqrt(e_tr)}
res["extra_test_rows_ge_1e6_if_untruncated"] = extra_rows
res["test_truncation_check"] = chk

# 2. later submissions
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06"]).set_index("id")
subs = {}
for name in ("sub_v19vce", "sub_v19ncapp", "sub_v19nca2pp", "BEST_CURRENT"):
    s = pd.read_csv(f"submissions/{name}.csv", usecols=["id", GAS, EUI])
    At = te.loc[s["id"], "bld_dim_06"].to_numpy(float)
    g, e = s[GAS].to_numpy(float), s[EUI].to_numpy(float)
    subs[name] = {"gas_max": float(g.max()), "n_gas_gt_1e6": int((g > C).sum()),
                  "n_eui_gt_1e6_over_A": int((e * At > C).sum()), "max_eui_times_A": float((e * At).max())}
res["submissions"] = subs
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
