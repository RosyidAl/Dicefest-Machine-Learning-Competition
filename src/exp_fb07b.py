"""FB07b A/B: extend the kNN-target block (FB07) with the neighbour spaces / windows that the residual correction
found useful (src/resid_knn.py CONFIGS): building era space, building + census PCA space, a 3-feature size space,
a very local (W=10) and a wide (W=1000) window in the full space.

Hypothesis: the residual correction still gains about 0.0006 total on top of the v16 blend, so the models miss
neighbourhood signal; giving it to the models as cross-fitted features captures it non-linearly.
A: XGB gpm (x4 params, 5-fold) with FB07 (= xk_ inputs). B: A + FB07b. Same folds, same code path.
usage: python src/exp_fb07b.py base|ext [--smoke]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import TARGET_COLS, load, nrmse  # noqa: E402
from fb07b import DataP, DataX  # noqa: E402
from resid_knn import spaces  # noqa: E402
from train_lib import run  # noqa: E402

VARIANT = sys.argv[1]
SMOKE = "--smoke" in sys.argv
tr, te = load()
DataPX = type("DataPX", (DataX, DataP), {})  # FB07 computed in parallel blocks (same code for both variants)
d = DataPX(n_folds=5) if VARIANT == "ext" else DataP(n_folds=5)
if VARIANT == "ext":
    S = spaces(tr, te)
    n = len(tr)
    d.sp = {k: (v[:n], v[n:]) for k, v in S.items()}
d.knn_cache_dir = "cache/knn_fb07b_p"  # FB07 part, shared by both variants
d.knn_x_cache_dir = "cache/knn_fb07b_x"
ex, ext = d.agg_features()
x4 = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
if SMOKE:
    d.folds = d.folds[:1]
    x4.update(learning_rate=0.3)
out = "exp/fb07b_smoke" if SMOKE else "exp/fb07b"
t0 = time.time()
oof, _ = run(d, f"ab_{VARIANT}_gpm", "xgb", "gpm", x4, use_te=True, windows=(2, 10), cells=True, knn=True,
             out_dir=out, wmul=1 / d.A, extra=ex, extra_t=ext)
res = {"variant": VARIANT, "minutes": round((time.time() - t0) / 60, 1)}
for m, col, mult in (("gas", TARGET_COLS[0], d.A), ("eui", TARGET_COLS[2], 1.0)):
    res[m] = [round(nrmse(d.y[col][b], (oof * mult)[b] if np.ndim(mult) == 0 else oof[b] * mult[b], col), 5)
              for _, b in d.folds]
print(json.dumps(res), flush=True)
if not SMOKE:
    with open(f"{out}/results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
