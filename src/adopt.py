"""Electric-heating adoption.

The target is a census (subzone-level) quantity: it is constant inside every subzone, and
adoption = (100 - sum(hvac_dist_*)) - r, where r is the share of the one heating category
missing from hvac_dist_*. Rows of test subzones present in train copy the train value;
for unseen subzones we model r at subzone level.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import HVAC, SEED, base_features, constant_cols, load  # noqa: E402
from te import group_stats, window_stats  # noqa: E402

LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED,
                  deterministic=True, force_row_wise=True, num_threads=3, verbose=-1)
WINDOWS = (1, 3, 10)


def subzone_table(df, X, oa_cols, pc_cols):
    X = X.copy()
    X["subzone_id"] = df["subzone_id"].values
    g = X.groupby("subzone_id", sort=False)
    S = g[oa_cols].first().join(g[pc_cols].mean().add_suffix("_m"))
    S["n_rows"] = g.size()
    S["sub_n"] = g["sub_n"].first()
    S["zone"] = df.groupby("subzone_id", sort=False)["zone_region_id"].first()
    return S


def neighbour_feats(fit_S, fit_r, app_S, size):
    """Zone-mean and windowed (adjacent subzone id) means of r from fit subzones."""
    out = {}
    s, c = group_stats(fit_S["zone"].values, fit_r[:, None], app_S["zone"].values)
    out["zn_r"] = np.where(c > 0, s[:, 0] / np.maximum(c, 1), np.nan)
    for w in WINDOWS:
        s, c = window_stats(fit_S["sub_n"].values.astype(int), fit_r[:, None], app_S["sub_n"].values.astype(int), w, size)
        out[f"w{w}_r"] = np.where(c > 0, s[:, 0] / np.maximum(c, 1), np.nan)
        out[f"w{w}_c"] = c
    return pd.DataFrame(out, index=app_S.index)


def fit_adoption(cache="cache", use_neighbours=True, verbose=True):
    tr, te = load(cache)
    const = constant_cols(tr)
    X, Xt = base_features(tr, const), base_features(te, const)
    nun = X.assign(subzone_id=tr["subzone_id"].values).groupby("subzone_id").nunique().max()
    oa_cols = [c for c in X.columns if nun[c] <= 1 and c not in ("sub_n", "zone_n")] + ["zone_n"]
    pc_cols = [c for c in X.columns if nun[c] > 1 and c != "sub_n"]
    S = subzone_table(tr, X, oa_cols, pc_cols)
    St = subzone_table(te, Xt, oa_cols, pc_cols)
    St = St[~St.index.isin(S.index)]  # only subzones unseen in train need a model
    y = tr.groupby("subzone_id", sort=False)["target_electric_heating_adoption"].first().reindex(S.index).values
    r = S["hvac_rest"].values - y
    size = int(max(S["sub_n"].max(), St["sub_n"].max()))
    feats = [c for c in S.columns if c != "zone"]

    oof = np.zeros(len(S))
    pt = np.zeros(len(St))
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(S))
    for k, (a, b) in enumerate(folds):
        Sa, Sb = S.iloc[a], S.iloc[b]
        Fa, Fb, Ft = Sa[feats], Sb[feats], St[feats]
        if use_neighbours:
            # inner OOF neighbour features for the fitting subzones
            na = pd.concat([neighbour_feats(Sa.iloc[ia], r[a][ia], Sa.iloc[ib], size)
                            for ia, ib in KFold(5, shuffle=True, random_state=SEED).split(Sa)]).reindex(Sa.index)
            Fa = Fa.join(na)
            Fb = Fb.join(neighbour_feats(Sa, r[a], Sb, size))
            Ft = Ft.join(neighbour_feats(Sa, r[a], St, size))
        m = lgb.train(LGB_PARAMS, lgb.Dataset(Fa, r[a]), 3000, valid_sets=[lgb.Dataset(Fb, r[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(Fb, num_iteration=m.best_iteration)
        pt += m.predict(Ft, num_iteration=m.best_iteration) / len(folds)
        if verbose:
            print(f"  adoption fold {k} it={m.best_iteration}", flush=True)
    pred_oof = np.clip(S["hvac_rest"].values - np.clip(oof, 0, None), 0, 100)
    rmse = float(np.sqrt(np.mean((pred_oof - y) ** 2)))
    if verbose:
        print(f"adoption unseen-subzone CV rmse={rmse:.5f}", flush=True)

    # assemble the test column
    known = dict(zip(S.index, y))
    unseen = dict(zip(St.index, np.clip(St["hvac_rest"].values - np.clip(pt, 0, None), 0, 100)))
    sub = te["subzone_id"].values
    pred = np.array([known[s] if s in known else unseen[s] for s in sub])
    frac_known = float(np.mean([s in known for s in sub]))
    return pred, rmse, frac_known


if __name__ == "__main__":
    for nb in (False, True):
        pred, rmse, fk = fit_adoption(use_neighbours=nb)
        print(f"neighbours={nb}: rmse_unseen={rmse:.5f} frac_known={fk:.4f} "
              f"expected test NRMSE={np.sqrt((1 - fk) * rmse ** 2) / 7.18:.5f}")
        np.save(f"exp/pred/adopt_nb{int(nb)}.npy", pred)
