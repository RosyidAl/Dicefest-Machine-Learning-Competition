"""v18 candidates: v16 kNN members with the FB07b block added (src/exp_fb07b.py DataX: extra kNN-target statistics
in the era / building+census / size spaces and W=10 / W=1000 windows). Built only if the FB07b A/B wins.
full: 10-fold on all train rows -> exp/oof|pred; hold: same model on the 70% part P (10-fold inside P),
predictions for the 30% holdout H -> exp/hold/oof|pred (needed to refit the residual-correction weights).
usage: python src/exp_v18_knn2.py full|hold name1,name2,...
"""
import copy
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, TARGET_COLS, load  # noqa: E402
from fb07b import DataX  # noqa: E402
from resid_knn import spaces  # noqa: E402
from train_lib import run  # noqa: E402

mode, names = sys.argv[1], sys.argv[2].split(",")

tr, te = load()
d = DataX(n_folds=10)
S = spaces(tr, te)
n = len(tr)
d.sp = {k: (v[:n], v[n:]) for k, v in S.items()}
d.knn_cache_dir = "cache/knn"  # FB07 part: same cache as the v16 members
d.knn_x_cache_dir = "cache/knn_x"  # FB07b part
ex, ext = d.agg_features()
x8 = dict(learning_rate=0.03, max_depth=10, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
x4 = dict(x8, learning_rate=0.05, max_depth=8)
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
XP = dict(x4, objective="count:poisson", eval_metric="poisson-nloglik")  # as src/exp_v17_pois.py
MLP = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))  # plain MLP (n4f / n6 settings)
PLR = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44), emb=dict(k=24, sigma=0.05, d=24),
           epochs=12, fixed_ep=11)
JOBS = {  # name: backend, kind, params, weight code (same formulations as the v16 kNN members)
    "xk2_gpm": ("xgb", "gpm", x8, "iA"),
    "xk2_lgpm": ("xgb", "lgas_per:bld_dim_12", x8, "A"),
    "xk2_gpgea": ("xgb", "gas_per:bld_dim_03", x8, "iA"),
    "xk2_lepa": ("xgb", "lelec_per:bld_dim_06", x4, "M"),
    "lk2_gpm": ("lgb", "gpm", lp, "iA"),
    "lk2_lepa": ("lgb", "lelec_per:bld_dim_06", lp, "M"),
    "pk2_gpgea": ("nn", "gas_per:bld_dim_03", PLR, "iA"),
    "pk2_lepm": ("nn", "lepm", dict(PLR, emb=dict(k=24, sigma=0.02, d=48), epochs=10, fixed_ep=9), "M"),
    "mk2": ("nn_multi", [("gpm", "gpm", "iA"), ("epm", "epm", "iM")], dict(PLR, hidden=(2048, 1024, 512)), None),
    # v19: Poisson objective (src/exp_v17_pois.py settings) with the FB07b inputs, and the PLR log-gas member
    "xp2_gpm": ("xgb", "gpm", dict(XP, min_child_weight=1e6), "iM"),
    "xp2_eui": ("xgb", "gas_per:bld_dim_06", dict(XP, min_child_weight=1.35e4), "iA"),
    "xp2_epm": ("xgb", "epm", dict(XP, min_child_weight=2.8e5), "iM"),
    "xp2_gpgea": ("xgb", "gas_per:bld_dim_03", dict(XP, min_child_weight=8e3), "iA"),
    "xp2_epa": ("xgb", "epa", dict(XP, min_child_weight=3.7e3), "iA"),
    "xp82_gpm": ("xgb", "gpm", dict(x8, objective="count:poisson", eval_metric="poisson-nloglik", min_child_weight=1e6), "iM"),
    "lp2_gpm": ("lgb", "gpm", dict(lp, objective="poisson"), "iM"),
    "lp2_epm": ("lgb", "epm", dict(lp, objective="poisson"), "iM"),
    "pk2_lgpm": ("nn", "lgas_per:bld_dim_12", PLR, "A"),
    # v20: kNN + FB07b versions of the blend members that had no kNN inputs (same kind / weight / params)
    "xk2_gpn": ("xgb", "gas_per:bld_dim_01", dict(x8, learning_rate=0.05), "iA"),  # x8_gpn
    "xk2_eui_wsM": ("xgb", "eui", x4, "sqrtM"),  # xgb_eui_wsM
    "xk2_gas": ("xgb", "gas", x4, None),  # xgb_gas
    "xk2_leui_wA": ("xgb", "leui", x4, "A"),  # xgb_leui_wA
    "xk2_elec": ("xgb", "elec", x4, None),  # xgb_elec
    "lk2_gas": ("lgb", "gas", lp, None),  # la_gas
    "lk2_leui_wA": ("lgb", "leui", lp, "A"),  # la_leui_wA
    "lk2_epa": ("lgb", "elec_per:bld_dim_06", lp, "iM"),  # l4_epa
    "nk2_gpm": ("nn", "gpm", dict(MLP, epochs=60, fixed_ep=37), "iA"),  # n4f_gpm
    "nk2_lgpm": ("nn", "lgas_per:bld_dim_12", dict(MLP, epochs=40, fixed_ep=39), "A"),  # n6_lgpm
    "nk2b_lepm": ("nn", "lepm", dict(MLP, epochs=24, fixed_ep=23, seeds=(45, 46, 47)), "M"),  # n6b_lepm
    "pk2_gpm": ("nn", "gpm", PLR, "iA"),  # p1_gpm formulation, PLR epochs of the kNN members
    "tk2_lepm": ("nn", "lepm", dict(lr=2e-3, wd=1e-4, drop=0.1, hidden=(512, 512, 512), seeds=(42,),
                                    emb=dict(k=24, sigma=0.05, d=24), ens=16, epochs=24, fixed_ep=23), "M"),  # t1_lepm
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
    s.sp = {k: (v[0][P], v[0][H]) for k, v in d.sp.items()}
    s.knn_cache_dir = "cache/knn"
    s.knn_x_cache_dir = "cache/knn_x"
    exs, exts = ex.iloc[P].reset_index(drop=True), ex.iloc[H].reset_index(drop=True)
    D, out, agg = s, "exp/hold", dict(extra=exs, extra_t=exts)
else:
    D, out, agg = d, "exp", dict(extra=ex, extra_t=ext)
W = {"A": D.A, "iA": 1.0 / D.A, "M": D.M.astype(float), "iM": 1.0 / D.M, "sqrtM": np.sqrt(D.M), None: None}
for name in names:
    backend, kind, params, wk = JOBS[name]
    if os.path.exists(f"{out}/pred/{name}.npy") or os.path.exists(f"{out}/pred/{name}_gpm.npy"):
        continue
    if backend == "nn_multi":
        from nn import run_nn_multi
        run_nn_multi(D, name, [(sfx, k, W[w]) for sfx, k, w in kind], out_dir=out, use_te=True, windows=(2, 10),
                     cells=True, knn=True, **agg, **params)
        continue
    if backend == "nn":
        from nn import run_nn
        run_nn(D, name, kind, out_dir=out, wmul=W[wk], use_te=True, windows=(2, 10), cells=True, knn=True, **agg,
               **params)
    else:
        run(D, name, backend, kind, params, wmul=W[wk], use_te=True, windows=(2, 10), cells=True, out_dir=out,
            knn=True, **agg)
