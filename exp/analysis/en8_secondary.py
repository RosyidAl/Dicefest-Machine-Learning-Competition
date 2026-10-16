"""EN8: ensemble robustness on CV_SECONDARY (zone-grouped folds). Retrains three heavily weighted v15nc members on
data/folds/folds_secondary_k5_s42.parquet (part=gpu: xa_gpgea XGB + n6_lgpm MLP; part=cpu: la_lepa LGB), and compares
each with its CV_PRIMARY OOF score; part=blend fits the NNLS blend of the available secondary members.
usage: python exp/analysis/en8_secondary.py gpu|cpu|blend"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

part = sys.argv[1]
OUT = "exp/secondary"
os.makedirs(f"{OUT}/oof", exist_ok=True)
os.makedirs(f"{OUT}/pred", exist_ok=True)
if part in ("gpu", "cpu"):
    from nn import run_nn
    from train_lib import Data, run
    d = Data(n_folds=10)
    ex, ext = d.agg_features()
    fold = pd.read_parquet("data/folds/folds_secondary_k5_s42.parquet")["fold"].values
    d.folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]
    TEC = dict(use_te=True, windows=(2, 10), cells=True)
    if part == "gpu":
        x8 = dict(learning_rate=0.03, max_depth=10, subsample=0.8, colsample_bytree=0.6, min_child_weight=50,
                  reg_lambda=1.0, eval_metric="rmse")
        run(d, "xa_gpgea", "xgb", "gas_per:bld_dim_03", x8, wmul=1.0 / d.A, out_dir=OUT, extra=ex, extra_t=ext, **TEC)
        run_nn(d, "n6_lgpm", "lgas_per:bld_dim_12", out_dir=OUT, wmul=d.A, epochs=40, fixed_ep=39, lr=1e-3, wd=1e-4,
               drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44), **TEC)
    else:
        lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
        run(d, "la_lepa", "lgb", "lelec_per:bld_dim_06", lp, wmul=d.M.astype(float), out_dir=OUT, extra=ex,
            extra_t=ext, **TEC)
else:
    tr = pd.read_parquet("data/cache/train.parquet")
    A = tr["bld_dim_06"].values
    for m, col, scale in (("xa_gpgea", TARGET_COLS[2], 1), ("n6_lgpm", TARGET_COLS[2], 1), ("la_lepa", TARGET_COLS[1], 1)):
        for src in ("exp/oof", f"{OUT}/oof"):
            p = np.load(f"{src}/{m}.npy")
            y = tr[col].values
            s = np.sqrt(np.mean((p - y) ** 2)) / SIGMAS[col]
            extra = ""
            if col == TARGET_COLS[2]:
                extra = f" gas {np.sqrt(np.mean((p * A - tr[TARGET_COLS[0]].values) ** 2)) / SIGMAS[TARGET_COLS[0]]:.5f}"
            print(f"{m:10s} {'primary' if src == 'exp/oof' else 'secondary':9s} {col.replace('target_', '')} {s:.5f}{extra}")
    from blend import cv_blend_score
    for src in ("exp/oof", f"{OUT}/oof"):
        P = np.column_stack([np.load(f"{src}/{m}.npy") for m in ("xa_gpgea", "n6_lgpm")])
        e = np.sqrt(np.mean((cv_blend_score(P, tr[TARGET_COLS[2]].values) - tr[TARGET_COLS[2]].values) ** 2)) / SIGMAS[TARGET_COLS[2]]
        print(f"blend xa_gpgea + n6_lgpm EUI ({'primary' if src == 'exp/oof' else 'secondary'}): {e:.5f}")
