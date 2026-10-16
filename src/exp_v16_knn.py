"""v16: members with FB07 kNN target encodings (Data.knn_features) on top of aggregates + target encodings + climate
cells. full: 10-fold on all train rows -> exp/oof|pred; hold: same model on the 70% part P (10-fold inside P),
predictions for the 30% holdout H -> exp/hold/oof|pred (needed to refit the residual-correction weights).

A/B (configs E0190 vs E0191, XGB gpm 5-fold): see reports/JOURNAL.md.
usage: python src/exp_v16_knn.py full|hold name1,name2,...
"""
import copy
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, TARGET_COLS  # noqa: E402
from train_lib import Data, run  # noqa: E402

mode, names = sys.argv[1], sys.argv[2].split(",")
d = Data(n_folds=10)
d.knn_cache_dir = "cache/knn"  # fold kNN encodings are shared by all members of the same split
ex, ext = d.agg_features()
x8 = dict(learning_rate=0.03, max_depth=10, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
x4 = dict(x8, learning_rate=0.05, max_depth=8)
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
JOBS = {  # name: backend, kind, params, weight code
    "xk_gpm": ("xgb", "gpm", x8, "iA"),
    "xk_lgpm": ("xgb", "lgas_per:bld_dim_12", x8, "A"),
    "xk_gpgea": ("xgb", "gas_per:bld_dim_03", x8, "iA"),
    "xk_lepa": ("xgb", "lelec_per:bld_dim_06", x4, "M"),
    "xk_epm_wM": ("xgb", "epm", x4, "iM"),
    "lk_gpm": ("lgb", "gpm", lp, "iA"),
    "lk_lepa": ("lgb", "lelec_per:bld_dim_06", lp, "M"),
    # PLR MLPs (p2 settings) with the kNN encodings
    "pk_gpm": ("nn", "gpm", dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44),
                                 emb=dict(k=24, sigma=0.05, d=24), epochs=12, fixed_ep=11), "iA"),
    "pk_lepm": ("nn", "lepm", dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44),
                                   emb=dict(k=24, sigma=0.02, d=48), epochs=10, fixed_ep=9), "M"),
}
if mode == "hold":
    perm = np.random.default_rng(7).permutation(len(d.X))
    P, H = np.sort(perm[: int(0.7 * len(perm))]), np.sort(perm[int(0.7 * len(perm)):])
    s = copy.copy(d)
    s.X, s.Xt = d.X.iloc[P].reset_index(drop=True), d.X.iloc[H].reset_index(drop=True)
    s.A, s.At, s.M, s.Mt = d.A[P], d.A[H], d.M[P], d.M[H]
    s.y = {c: d.y[c][P] for c in TARGET_COLS}
    s.sub, s.subt, s.zone, s.zonet = d.sub[P], d.sub[H], d.zone[P], d.zone[H]
    s.subn, s.subnt = d.subn[P], d.subn[H]
    s.V, s.W = d.V[P], d.W[P]
    s.cells, s.cellst = [c[P] for c in d.cells], [c[H] for c in d.cells]
    s.folds = list(KFold(10, shuffle=True, random_state=SEED).split(s.X))
    s._kz = None
    s.knn_cache_dir = "cache/knn"
    exs, exts = ex.iloc[P].reset_index(drop=True), ex.iloc[H].reset_index(drop=True)
    D, out, agg = s, "exp/hold", dict(extra=exs, extra_t=exts)
else:
    D, out, agg = d, "exp", dict(extra=ex, extra_t=ext)
W = {"A": D.A, "iA": 1.0 / D.A, "M": D.M.astype(float), "iM": 1.0 / D.M}
for name in names:
    backend, kind, params, wk = JOBS[name]
    if os.path.exists(f"{out}/pred/{name}.npy"):
        continue
    if backend == "nn":
        from nn import run_nn
        run_nn(D, name, kind, out_dir=out, wmul=W[wk], use_te=True, windows=(2, 10), cells=True, knn=True, **agg,
               **params)
    else:
        run(D, name, backend, kind, params, wmul=W[wk], use_te=True, windows=(2, 10), cells=True, out_dir=out,
            knn=True, **agg)
