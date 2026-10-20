"""Audit postproc 01: train target distributions (bounds, truncation, rounding grid) and a scan of the submitted
file sub_v19vce.csv against them. Light: reads only the needed parquet columns.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/01_dist.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
cols = ["id", "bld_dim_01", "bld_dim_02", "bld_dim_06", "bld_dim_12", "subzone_id"] + [f"hvac_dist_{i:02d}" for i in range(1, 11)]
tr = pd.read_parquet("cache/train.parquet", columns=cols + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=cols)
sub = pd.read_csv("submissions/sub_v19vce.csv")
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
out = {}


def q(x, ps=(0, 1e-5, 1e-4, 1e-3, 0.01, 0.5, 0.99, 0.999, 0.9999, 0.99999, 1)):
    x = np.asarray(x, float)
    return {str(p): float(np.quantile(x, p)) for p in ps}


A, M, E = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float), tr["bld_dim_01"].values.astype(float)
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
y = {c: tr[c].values.astype(float) for c in TARGET_COLS}
derived = {"gas": y[GAS], "elec": y[EL], "eui": y[EUI], "gas_per_meter": y[GAS] / M, "elec_per_meter": y[EL] / M,
           "elec_per_area": y[EL] / A, "clean": y[CL], "adopt": y[AD], "A": A, "M": M}
for k, v in derived.items():
    out[f"train_{k}"] = dict(q(v), n_zero=int((v == 0).sum()), n_nan=int(np.isnan(v).sum()), mean=float(np.nanmean(v)))
# decimal grid of the targets: fraction of values that are integer / 1-dec / 2-dec
for c in TARGET_COLS:
    v = y[c]
    out[f"grid_{c}"] = {d: float(np.mean(np.abs(np.round(v, d) - v) < 1e-9)) for d in (0, 1, 2, 3, 4)}
# identities
cl = np.round(100 * tr["bld_dim_02"].values / tr["bld_dim_01"].values, 2)
out["clean_identity_exact"] = int((cl == y[CL]).sum()), int(len(cl))
out["clean_identity_maxabs"] = float(np.nanmax(np.abs(cl - y[CL])))
out["eui_identity_max_rel"] = float(np.max(np.abs(y[EUI] * A - y[GAS]) / np.maximum(y[GAS], 1)))
out["train_b01_zero"], out["test_b01_zero"] = int((E == 0).sum()), int((te["bld_dim_01"].values == 0).sum())
# submission scan
assert (sub["id"].values == ss["id"].values).all() and list(sub.columns) == ["id"] + TARGET_COLS
out["sub_rows"] = len(sub)
for c in TARGET_COLS:
    v = sub[c].values.astype(float)
    lo, hi = y[c].min(), y[c].max()
    out[f"sub_{c}"] = dict(q(v), n_nan=int(np.isnan(v).sum()), n_neg=int((v < 0).sum()), n_zero=int((v == 0).sum()),
                         n_below_train_min=int((v < lo).sum()), n_above_train_max=int((v > hi).sum()),
                         train_min=float(lo), train_max=float(hi))
# per-row consistency of submitted gas and EUI: gas - EUI * A (relative)
g, e = sub[GAS].values, sub[EUI].values
rel = (g - e * At) / np.maximum(g, 1)
out["sub_gas_vs_eui_A_rel"] = q(rel)
out["sub_gas_vs_eui_A_rms_over_sigma_gas"] = float(np.sqrt(np.mean((g - e * At) ** 2)) / SIGMAS[GAS])
gm, em = g / Mt, sub[EL].values / Mt
out["sub_gas_per_meter"] = q(gm)
out["sub_elec_per_meter"] = q(em)
for k, v, ref in (("gas_per_meter", gm, derived["gas_per_meter"]), ("elec_per_meter", em, derived["elec_per_meter"])):
    out[f"sub_{k}_outside_train"] = dict(below=int((v < ref.min()).sum()), above=int((v > ref.max()).sum()))
out["sub_eui_above_trainmaxgas_over_A"] = int((e > y[GAS].max() / At).sum())
# clean ratio in submission vs identity
clt = np.round(100 * te["bld_dim_02"].values / te["bld_dim_01"].values, 2)
out["sub_clean_equals_identity"] = int((np.abs(sub[CL].values - clt) < 1e-9).sum())
# adoption bound: adoption <= rest = 100 - sum(hvac) (r >= 0), check on train and on the submission
H = [f"hvac_dist_{i:02d}" for i in range(1, 11)]
rest_tr = 100 - tr[H].sum(1).values
rest_te = 100 - te[H].sum(1).values
out["train_adopt_minus_rest"] = q(y[AD] - rest_tr)
out["sub_adopt_minus_rest"] = q(sub[AD].values - rest_te)
out["sub_adopt_above_rest"] = int((sub[AD].values > rest_te + 1e-9).sum())
json.dump(out, open("exp/audit/postproc/01_dist.json", "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v)[:600])
