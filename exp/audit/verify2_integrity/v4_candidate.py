"""verify2_integrity / step 4: does the candidate file implement the claimed shrink exactly, and only that?
Re-applies the shrink (own implementation, claimed P-fitted parameters) to submissions/sub_v19vce.csv on the
non-duplicate rows and compares with exp/audit/integrity/sub_v19vce_trunc2.csv.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/v4_candidate.py
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "final/src")
from dup_fix import KEY, duplicate_targets  # noqa: E402
from common import TARGET_COLS  # noqa: E402

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
CL = dict(a0=0.008064793880232542, b0=0.3929692979589114, k=0.7, c_gas=1.0126184473794142, c_eui=1.141696475118523)
U = 1e6
CAND = "exp/audit/integrity/sub_v19vce_trunc2.csv"


def tfun(pg, Mv, a0, b0, k):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / Mv, 1e-6)), 1.0)
    a = (U - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


tr = pd.read_parquet("cache/train.parquet", columns=sorted(set(KEY + TARGET_COLS)))
te = pd.read_parquet("cache/test.parquet", columns=sorted(set(KEY + ["id"])))
dup = duplicate_targets(tr, te)
hit = ~np.isnan(dup[:, 0])
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
s0 = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
s1 = pd.read_csv(CAND, float_precision="round_trip")
res = dict(n_dup=int(hit.sum()), sha_cand=hashlib.sha256(open(CAND, "rb").read()).hexdigest())
assert (s0["id"].values == te["id"].values).all() and (s1["id"].values == s0["id"].values).all()
g0, e0 = s0[GAS].values, s0[EUI].values
g1 = np.clip(g0 + CL["c_gas"] * tfun(g0, Mt, CL["a0"], CL["b0"], CL["k"]), 0, None)
e1 = np.clip((e0 * At + CL["c_eui"] * tfun(e0 * At, Mt, CL["a0"], CL["b0"], CL["k"])) / At, 0, None)
g1[hit], e1[hit] = g0[hit], e0[hit]
res["max_abs_diff_gas_vs_cand"] = float(np.max(np.abs(g1 - s1[GAS].values)))
res["max_abs_diff_eui_vs_cand"] = float(np.max(np.abs(e1 - s1[EUI].values)))
res["dup_rows_unchanged_in_cand"] = bool(all((s1.loc[hit, c].values == s0.loc[hit, c].values).all() for c in TARGET_COLS))
res["dup_rows_equal_train_targets_in_cand"] = bool(np.allclose(s1.loc[hit, TARGET_COLS].values, dup[hit], rtol=0, atol=1e-9))
res["other_cols_identical"] = {c: bool((s1[c].values == s0[c].values).all()) for c in (EL,) + tuple(TARGET_COLS[3:])}
# how much would the shrink move the duplicate rows if it were (wrongly) applied after the copy
gd = g0[hit]
res["dup_shift_if_applied_max"] = float(np.max(np.abs(CL["c_gas"] * tfun(gd, Mt[hit], CL["a0"], CL["b0"], CL["k"]))))
res["dup_gas_max"] = float(gd.max())
# size of the change on test
d = s1[GAS].values - g0
res["test_gas"] = dict(n_rel_gt_1pct=int((np.abs(d) > 0.01 * g0).sum()), min=float(d.min()), max=float(d.max()),
                       sum_sq_over_sigma2_per_row=float(np.mean(d ** 2) / 151750.04 ** 2),
                       n_pred_gt_9e5=int((g0 > 9e5).sum()), n_pred_gt_1e6=int((g0 > 1e6).sum()))
d = s1[EUI].values - e0
res["test_eui"] = dict(n_rel_gt_1pct=int((np.abs(d) > 0.01 * e0).sum()), min=float(d.min()), max=float(d.max()))
# shift as a function of prediction level on test (gas)
bands = [(8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 2e6)]
res["test_gas_mean_shift_by_band"] = {f"{int(a/1e3)}k-{int(b/1e3)}k": dict(n=int(((g0 >= a) & (g0 < b)).sum()),
                                      mean_shift=float(np.mean((s1[GAS].values - g0)[(g0 >= a) & (g0 < b)])))
                                      for a, b in bands}
json.dump(res, open("exp/audit/verify2_integrity/v4_candidate.json", "w"), indent=1)
print(json.dumps(res, indent=1))
