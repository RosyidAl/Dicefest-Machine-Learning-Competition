"""Fold-0 A/B (E0266-E0268): PLR MLP (pk_* settings, FB07 inputs) with MSE vs Poisson-deviance loss.
One seed, fold 0 of the 10-fold split used by the members; scores on the fold-0 rows only.
usage: python src/exp_nn_pois_ab.py name1,name2,...   (names from VARIANTS)"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import TARGET_COLS, nrmse  # noqa: E402
from nn import run_nn  # noqa: E402
from train_lib import Data  # noqa: E402

d = Data(n_folds=10)
d.knn_cache_dir = "cache/knn"
ex, ext = d.agg_features()
d.folds = d.folds[:1]
b = d.folds[0][1]
W = {"A": d.A, "iA": 1.0 / d.A, "M": d.M.astype(float), "iM": 1.0 / d.M}
P = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), emb=dict(k=24, sigma=0.05, d=24), epochs=12, fixed_ep=11)
VARIANTS = {  # name: kind, weight code, extra params
    "E0266_mse_gpm": ("gpm", "iA", {}),
    "E0267_pois_gpm": ("gpm", "iM", dict(loss="poisson")),
    "E0268_pois_eui": ("gas_per:bld_dim_06", "iA", dict(loss="poisson")),
    "E0269_mse_epm": ("epm", "iM", {}),
    "E0270_pois_epm": ("epm", "iM", dict(loss="poisson")),
}
os.makedirs("exp/nnpois/oof", exist_ok=True); os.makedirs("exp/nnpois/pred", exist_ok=True)
for name in sys.argv[1].split(","):
    kind, wk, extra = VARIANTS[name]
    t0 = time.time()
    run_nn(d, name, kind, out_dir="exp/nnpois", wmul=W[wk], use_te=True, windows=(2, 10), cells=True, knn=True,
           extra=ex, extra_t=ext, seeds=(42,), **P, **extra)
    oof = np.load(f"exp/nnpois/oof/{name}.npy")
    if "epm" in kind:
        res = dict(elec=nrmse(d.y[TARGET_COLS[1]][b], oof[b], TARGET_COLS[1]))
    else:
        res = dict(eui=nrmse(d.y[TARGET_COLS[2]][b], oof[b], TARGET_COLS[2]),
                   gas=nrmse(d.y[TARGET_COLS[0]][b], oof[b] * d.A[b], TARGET_COLS[0]))
    res["min"] = round((time.time() - t0) / 60, 1)
    print("RESULT", name, json.dumps(res), flush=True)
    with open("exp/nnpois/results.jsonl", "a") as f:
        f.write(json.dumps(dict(name=name, **res)) + "\n")
