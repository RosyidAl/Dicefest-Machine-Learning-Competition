"""verify2_integrity / step 1: data facts behind the gas-truncation claim (independent code).
- train gas upper bound and density near 1e6 (and lower end), EUI*A == gas
- is the TEST set filtered the same way? compare upper-tail counts of the same predictor (global NNLS blend of the
  v19vce gas members: train OOF vs test fold-average) and of raw size features (A, M) against the 0.4286 row ratio
- candidate file vs sub_v19vce: which rows/columns move
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/v1_facts.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

OUT = "exp/audit/verify2_integrity/v1_facts.json"
GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
res = {}
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EL, EUI, "bld_dim_06", "bld_dim_12", "id"])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06", "bld_dim_12", "id"])
g, A, M = tr[GAS].values, tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
ratio = len(te) / len(tr)
res["n_train"], res["n_test"], res["row_ratio"] = len(tr), len(te), ratio
res["gas_max"], res["gas_min"] = float(g.max()), float(g.min())
res["n_gas_ge_1e6"] = int((g >= 1e6).sum())
edges = np.arange(800_000, 1_050_001, 25_000)
res["gas_hist_800k_1050k"] = {f"{int(a)}-{int(b)}": int(((g >= a) & (g < b)).sum()) for a, b in zip(edges[:-1], edges[1:])}
lo = np.arange(0, 30_001, 2_500)
res["gas_hist_low"] = {f"{int(a)}-{int(b)}": int(((g >= a) & (g < b)).sum()) for a, b in zip(lo[:-1], lo[1:])}
res["eui_times_A_max"] = float((tr[EUI].values * A).max())
res["eui_A_vs_gas_maxreldiff"] = float(np.max(np.abs(tr[EUI].values * A - g) / g))
e = tr[EL].values
eedges = np.quantile(e, [0.999, 0.9995, 0.9999, 1.0])
res["elec_top_quantiles"] = [float(x) for x in eedges]
# elec smoothness check near its max: counts in 10 equal bins over the top 0.1%
top = e[e >= np.quantile(e, 0.999)]
res["elec_top_hist"] = np.histogram(top, 10)[0].tolist()
# --- test filtering check with one predictor applied to both sides
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
Po = np.column_stack([np.load(f"exp/oof/{m}.npy") for m in mem])
Pt = np.column_stack([np.load(f"exp/pred/{m}.npy") for m in mem])
w, _ = nnls(Po * A[:, None], tr[EUI].values * A)  # gas-scale NNLS (as nnls_blend with row_scale=A)
pf, pt = (Po @ w) * A, (Pt @ w) * At
del Po, Pt
res["global_nnls_oof_nrmse_gas"] = float(np.sqrt(np.mean((g - pf) ** 2)) / 151750.04)
tail = {}
for t in (7e5, 8e5, 9e5, 9.5e5, 1e6, 1.05e6, 1.1e6):
    ntr, nte = int((pf > t).sum()), int((pt > t).sum())
    tail[f">{int(t)}"] = dict(train_oof=ntr, test=nte, test_over_expected=round(nte / max(ntr * ratio, 1e-9), 3))
res["pred_tail_counts"] = tail
# true-y tail vs predicted tail on train: how many rows predicted > t have y near the bound
res["train_rows_pred_gt_1e6_mean_y"] = float(g[pf > 1e6].mean()) if (pf > 1e6).any() else None
feat = {}
for name, a, at in (("A", A, At), ("M", M, Mt), ("AxM", A * M, At * Mt)):
    for q in (0.99, 0.999, 0.9999):
        t = np.quantile(a, q)
        ntr, nte = int((a > t).sum()), int((at > t).sum())
        feat[f"{name}>q{q}"] = dict(train=ntr, test=nte, test_over_expected=round(nte / (ntr * ratio), 3))
res["feature_tail_counts"] = feat
# --- candidate vs current best submission
s0 = pd.read_csv("submissions/sub_v19vce.csv")
s1 = pd.read_csv("exp/audit/integrity/sub_v19vce_trunc2.csv")
assert (s0["id"].values == s1["id"].values).all()
cmp = {}
for c in s0.columns[1:]:
    d = s1[c].values - s0[c].values
    cmp[c] = dict(n_changed=int((d != 0).sum()), n_rel_gt_1pct=int((np.abs(d) > 0.01 * np.abs(s0[c].values)).sum()),
                  min_diff=float(d.min()), max_diff=float(d.max()), rms_over_sigma=None)
for c, sg in ((GAS, 151750.04), (EUI, 40.65)):
    d = s1[c].values - s0[c].values
    cmp[c]["rms_over_sigma"] = float(np.sqrt(np.mean(d ** 2)) / sg)
res["candidate_vs_v19vce"] = cmp
res["sub_gas_max"] = float(s0[GAS].max())
res["sub_n_gas_gt_1e6"] = int((s0[GAS] > 1e6).sum())
res["sub_n_gas_gt_9.5e5"] = int((s0[GAS] > 9.5e5).sum())
res["cand_n_gas_gt_1e6"] = int((s1[GAS] > 1e6).sum())
res["sub_n_euiA_gt_1e6"] = int((s0[EUI].values * At > 1e6).sum())
res["cand_n_euiA_gt_1e6"] = int((s1[EUI].values * At > 1e6).sum())
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=0)[:6000])
