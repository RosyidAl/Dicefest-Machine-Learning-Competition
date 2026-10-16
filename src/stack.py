"""Level-2 stacking: small LightGBM on level-1 OOF predictions plus scale features.

Gas-family stack predicts EUI; trained once with unit weights (EUI metric) and once with
weights A^2 (gas metric, since gas error = A * EUI error).
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, TARGET_COLS, load, nrmse  # noqa: E402

L2_PARAMS = dict(objective="regression", learning_rate=0.02, num_leaves=15, min_data_in_leaf=500,
                 feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=SEED,
                 deterministic=True, force_row_wise=True, num_threads=8, verbose=-1)


def stack_features(P, A, M, n):
    F = pd.DataFrame(P, columns=[f"m{i}" for i in range(P.shape[1])])
    F["mean"] = P.mean(1)
    F["spread"] = P.std(1)
    F["lA"] = np.log(A)
    F["lM"] = np.log(M)
    F["ln"] = np.log(n)
    return F


def fit_stack(F, y, Ft, w=None, seed=SEED + 1):
    """CV on a different fold split than level 1 (seed+1) to keep level-2 OOF honest-ish."""
    oof = np.zeros(len(F))
    pt = np.zeros(len(Ft))
    folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    for a, b in folds:
        da = lgb.Dataset(F.iloc[a], y[a], weight=None if w is None else w[a])
        db = lgb.Dataset(F.iloc[b], y[b], weight=None if w is None else w[b], reference=da)
        m = lgb.train(L2_PARAMS, da, 5000, valid_sets=[db], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict(F.iloc[b], num_iteration=m.best_iteration)
        pt += m.predict(Ft, num_iteration=m.best_iteration) / len(folds)
    return oof, pt


def stack_gas_family(runs, d="exp"):
    tr, te = load()
    A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
    M, Mt = tr["bld_dim_12"].values, te["bld_dim_12"].values
    n, nt = tr["bld_dim_01"].values, te["bld_dim_01"].values
    P = np.column_stack([np.load(f"{d}/oof/{r}.npy") for r in runs])
    Pt = np.column_stack([np.load(f"{d}/pred/{r}.npy") for r in runs])
    F, Ft = stack_features(P, A, M, n), stack_features(Pt, At, Mt, nt)
    y = tr[TARGET_COLS[2]].values
    o_eui, t_eui = fit_stack(F, y, Ft)
    o_gas, t_gas = fit_stack(F, y, Ft, w=A ** 2 / np.mean(A ** 2))
    print("stack eui:", nrmse(y, o_eui, TARGET_COLS[2]), " gas:", nrmse(tr[TARGET_COLS[0]].values, o_gas * A, TARGET_COLS[0]))
    return (o_eui, t_eui), (o_gas, t_gas)


def stack_elec_family(runs, d="exp"):
    tr, te = load()
    A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
    M, Mt = tr["bld_dim_12"].values, te["bld_dim_12"].values
    n, nt = tr["bld_dim_01"].values, te["bld_dim_01"].values
    # stack on the per-meter scale so the trees see comparable magnitudes, weight M^2 = elec MSE
    P = np.column_stack([np.load(f"{d}/oof/{r}.npy") for r in runs]) / M[:, None]
    Pt = np.column_stack([np.load(f"{d}/pred/{r}.npy") for r in runs]) / Mt[:, None]
    F, Ft = stack_features(P, A, M, n), stack_features(Pt, At, Mt, nt)
    y = tr[TARGET_COLS[1]].values / M
    o, t = fit_stack(F, y, Ft, w=M ** 2 / np.mean(M ** 2))
    print("stack elec:", nrmse(tr[TARGET_COLS[1]].values, o * M, TARGET_COLS[1]))
    return o * M, t * Mt


if __name__ == "__main__":
    gas_runs = sys.argv[1].split(",")
    elec_runs = sys.argv[2].split(",")
    stack_gas_family(gas_runs)
    stack_elec_family(elec_runs)
