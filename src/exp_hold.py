"""Holdout blending data: retrain blend members on a fixed 70% of train (10-fold CV inside it) and save
fold-averaged predictions for the other 30%. Those holdout predictions behave like test predictions
(averages over fold models), unlike OOF predictions, so blend weights can be fitted on them.

usage: python src/exp_hold.py gpu|nn|cpu   (gpu = CatBoost/XGBoost, nn = MLPs, cpu = LightGBM)
"""
import sys; sys.path.insert(0, "src")
import copy
import os
import numpy as np
from sklearn.model_selection import KFold
from common import SEED, TARGET_COLS
from train_lib import Data, run
from nn import run_nn

d = Data(n_folds=10)
perm = np.random.default_rng(7).permutation(len(d.X))
P, H = np.sort(perm[: int(0.7 * len(perm))]), np.sort(perm[int(0.7 * len(perm)):])


def subset(d, P, H, k=10):
    s = copy.copy(d)
    s.X, s.Xt = d.X.iloc[P].reset_index(drop=True), d.X.iloc[H].reset_index(drop=True)
    s.A, s.At, s.M, s.Mt = d.A[P], d.A[H], d.M[P], d.M[H]
    s.y = {c: d.y[c][P] for c in TARGET_COLS}
    s.sub, s.subt, s.zone, s.zonet = d.sub[P], d.sub[H], d.zone[P], d.zone[H]
    s.subn, s.subnt = d.subn[P], d.subn[H]
    s.V, s.W = d.V[P], d.W[P]
    s.cells, s.cellst = [c[P] for c in d.cells], [c[H] for c in d.cells]
    s.folds = list(KFold(k, shuffle=True, random_state=SEED).split(s.X))
    return s


s = subset(d, P, H)
A, M = s.A, s.M.astype(float)
W = {"A": A, "iA": 1.0 / A, "M": M, "iM": 1.0 / M, None: None}
cp = dict(learning_rate=0.08, depth=8, l2_leaf_reg=3, border_count=254)
xp = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
xf = dict(xp, learning_rate=0.08)
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
nk = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp/hold")
GPU = [  # name, backend, kind, params (or MLP epochs), weight
    ("c4_eui_wA", "cat", "eui", cp, "A"), ("n6_gpm", "nn", "gpm", 44, "iA"), ("n6_lepm", "nn", "lepm", 24, "M"),
    ("c4_lepa", "cat", "lelec_per:bld_dim_06", cp, "M"), ("x4_gpm", "xgb", "gpm", xp, "iA"),
    ("c4_lgpm", "cat", "lgas_per:bld_dim_12", cp, "A"), ("x4_lepa", "xgb", "lelec_per:bld_dim_06", xp, "M"),
    ("n6_epm_wM", "nn", "epm", 23, "iM"), ("c4_gpm", "cat", "gpm", cp, "iA"),
    ("c4_epa", "cat", "elec_per:bld_dim_06", cp, "iM"), ("x4_gpgea", "xgb", "gas_per:bld_dim_03", xp, "iA"),
    ("n6_gpgea", "nn", "gas_per:bld_dim_03", 36, "iA"), ("x4_elec", "xgb", "elec", xp, None),
    ("c4_leui_wA", "cat", "leui", cp, "A"), ("x4_epa", "xgb", "elec_per:bld_dim_06", xp, "iM"),
    ("x4_gpn", "xgb", "gas_per:bld_dim_01", xf, "iA"), ("c4_lepm", "cat", "lepm", cp, "M"),
    ("n6_lgpm", "nn", "lgas_per:bld_dim_12", 40, "A"), ("c4_epm_wM", "cat", "epm", cp, "iM"),
    ("c4_gpgea", "cat", "gas_per:bld_dim_03", cp, "iA"), ("n6_epa", "nn", "elec_per:bld_dim_06", 20, "iM"),
    ("x4_gas", "xgb", "gas", xp, None), ("n6_eui_wA", "nn", "eui", 36, "A"), ("x4_epm_wM", "xgb", "epm", xp, "iM"),
    ("n6_leui_wA", "nn", "leui", 33, "A"), ("x4_lepm", "xgb", "lepm", xp, "M"), ("x4_eui_wA", "xgb", "eui", xp, "A"),
    ("x4_lgpm", "xgb", "lgas_per:bld_dim_12", xp, "A"), ("x4_leui_wA", "xgb", "leui", xp, "A"),
]
CPU = [
    ("l4_gpm", "lgb", "gpm", lp, "iA"), ("l4_epa", "lgb", "elec_per:bld_dim_06", lp, "iM"),
    ("l4_gpgea", "lgb", "gas_per:bld_dim_03", lp, "iA"), ("l4_lepm", "lgb", "lepm", lp, "M"),
    ("l4_lgpm", "lgb", "lgas_per:bld_dim_12", lp, "A"), ("l4_elec", "lgb", "elec", lp, None),
    ("l4_leui_wA", "lgb", "leui", lp, "A"), ("l4_epm_wM", "lgb", "epm", lp, "iM"),
    ("l4_eui_wA", "lgb", "eui", lp, "A"), ("l4_lepa", "lgb", "lelec_per:bld_dim_06", lp, "M"),
]


def main():
    if sys.argv[1] == "xgbrev":
        # second GPU worker: XGBoost members from the end of the GPU list. An empty placeholder claims a member so
        # the forward "gpu" worker skips it; stop once the forward worker has reached the next member.
        fwd = [r for r in GPU if r[1] != "nn"]
        done = lambda nm: os.path.exists(f"exp/hold/pred/{nm}.npy")
        for i in range(len(fwd) - 1, -1, -1):
            name, backend, kind, params, wk = fwd[i]
            if backend != "xgb" or done(name):
                continue
            if next(j for j, r in enumerate(fwd) if not done(r[0])) >= i:
                break
            open(f"exp/hold/pred/{name}.npy", "wb").close()
            run(s, name, backend, kind, params, wmul=W[wk], verbose=False, **TE)
        return
    todo = {"gpu": [r for r in GPU if r[1] != "nn"], "nn": [r for r in GPU if r[1] == "nn"], "cpu": CPU}[sys.argv[1]]
    for name, backend, kind, params, wk in todo:
        if os.path.exists(f"exp/hold/pred/{name}.npy"):
            continue
        if backend == "nn":
            run_nn(s, name, kind, out_dir="exp/hold", wmul=W[wk], use_te=True, windows=(2, 10), cells=True,
                   epochs=params, fixed_ep=params - 1, **nk)
        else:
            run(s, name, backend, kind, params, wmul=W[wk], verbose=False, **TE)


if __name__ == "__main__":
    main()
