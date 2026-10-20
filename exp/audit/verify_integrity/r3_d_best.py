"""Verifier round 3 (leakage/optimism lens), step D: does BEST_CURRENT (sub_v19nca2pp, D015 post-processing) already
contain the truncation shrink? Compares sub_v19nca2 (before the audit post-processing) with sub_v19nca2pp near the gas
bound, and measures what the integrity shrink I (P-fitted, r3_a) would do if applied on top of BEST_CURRENT
(double application) or to the pre-post-processing file instead of S3.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/r3_d_best.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
OUT = "exp/audit/verify_integrity/r3_d_best.json"
r3a = json.load(open("exp/audit/verify_integrity/r3_a_rederive.json"))
a0, b0 = r3a["var_model_P"]["a0"], r3a["var_model_P"]["b0"]
KI = {m: r3a["fit_P"][m]["k"] for m in ("gas", "eui")}
CI = {m: r3a["fit_P"][m]["c"] for m in ("gas", "eui")}


def adj_I(pg, m, k):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / m, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


te = pd.read_parquet("data/cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
best = pd.read_csv("submissions/BEST_CURRENT.csv", float_precision="round_trip")
pre = pd.read_csv("submissions/sub_v19nca2.csv", float_precision="round_trip")
pp = pd.read_csv("submissions/sub_v19nca2pp.csv", float_precision="round_trip")
out = dict(best_sha=hashlib.sha256(open("submissions/BEST_CURRENT.csv", "rb").read()).hexdigest(),
           best_equals_nca2pp=bool(best.equals(pp)))
assert (pre["id"].values == te["id"].values).all() and (pp["id"].values == te["id"].values).all()
for m, col in (("gas", GAS), ("eui", EUI)):
    sc = At if m == "eui" else 1.0
    p0, p1 = pre[col].values, pp[col].values
    g0 = p0 * sc
    r = {}
    for lo, hi in ((8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 1e7)):
        g = (g0 >= lo) & (g0 < hi)
        dI = CI[m] * adj_I(g0[g], Mt[g], KI[m]) / (sc[g] if m == "eui" else 1.0)
        r[f"pre_{lo:.0f}-{hi:.0f}"] = dict(n=int(g.sum()), mean_change_pp_minus_pre=float((p1 - p0)[g].mean()),
                                           mean_change_I_on_pre=float(dI.mean()))
    dI_pre = CI[m] * adj_I(g0, Mt, KI[m]) / sc
    dS = p1 - p0
    near = g0 >= 8e5
    r["corr_S1S3S4_vs_I_near_bound"] = float(np.corrcoef(dS[near], dI_pre[near])[0, 1])
    r["max_after_pp_gas_scale"] = float((p1 * sc).max())
    r["n_pp_gas_scale_ge_1e6"] = int((p1 * sc >= 1e6).sum())
    dI_best = CI[m] * adj_I(p1 * sc, Mt, KI[m]) / sc
    r["I_on_best_rows_moved_gt_1pct"] = int((np.abs(dI_best) > 0.01 * p1).sum())
    r["I_on_best_rms_over_sigma"] = float(np.sqrt(np.mean(dI_best ** 2)) / SIG[m])
    r["I_on_best_min_change"] = float(dI_best.min())
    r["S_rms_over_sigma_near_bound_only"] = float(np.sqrt(np.mean(np.where(near, dS, 0) ** 2)) / SIG[m])
    out[m] = r
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
