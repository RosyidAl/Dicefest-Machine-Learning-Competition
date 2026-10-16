"""Electric-heating adoption, v2.

Census structure: in every subzone (census output area) the hvac_dist_* shares are exact counts / N, where N is the
number of households. The smallest N that makes every share integral recovers N exactly. The households outside the
listed categories number K = (100 - sum(hvac_dist_*)) * N / 100, and they split into electric heating (the target)
and one missing category r: adoption = 100 * (K - count_r) / N.

Rows of subzones present in train copy the train value (the target is constant within a subzone). For the ~8% of
test rows in unseen subzones, count_r is predicted at subzone level by averaging two LightGBM models:
- q_xent: share q = count_r / K, binomial cross-entropy with weight K;
- r_reg: r = 100 * count_r / N, squared error.
Features: the subzone's census columns, means of the postcode-level columns over its train + test rows, N, K, and
pooled count_r / K of neighbouring labelled subzones (zone, climate cell, sub_n windows).

Validation is 5-fold over train subzones, also reweighted to the row-count mix of the unseen test subzones (mostly
single-row subzones with large K). Unseen-weighted RMSE: old model 1.042, q_xent 1.033, r_reg 1.035, average 1.030.

usage: python src/adopt_v2.py   -> exp/pred/adopt_v2.npy
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import HVAC, SEED, base_features, constant_cols, load  # noqa: E402

BASE = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.5, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, seed=SEED, deterministic=True, force_row_wise=True, num_threads=2,
            verbose=-1)
WINDOWS = (1, 3, 10, 30)
# row counts of the test rows in unseen subzones, by the subzone's total row count (6 = 6 or more)
UNSEEN_MIX = {1: 5916, 2: 5158, 3: 2664, 4: 944, 5: 255, 6: 42}


def household_count(shares, n_max=1000, tol=2e-4):
    """Smallest N <= n_max with every share * N / 100 within tol of an integer (shares: (n, k) percentages)."""
    V = np.asarray(shares, dtype=float) / 100.0
    best = np.zeros(len(V), int)
    for N in range(1, n_max + 1):
        ok = (best == 0) & (np.abs(V * N - np.round(V * N)) < tol * max(1, N / 100)).all(1)
        best[ok] = N
    return best


def subzone_table(tr, te):
    const = constant_cols(tr)
    X = pd.concat([base_features(tr, const), base_features(te, const)], ignore_index=True)
    al = pd.concat([tr, te], ignore_index=True)
    X["subzone_id"] = al["subzone_id"].values
    nun = X.iloc[:len(tr)].groupby("subzone_id").nunique().max()
    oa = [c for c in X.columns if c != "subzone_id" and nun[c] <= 1 and c != "sub_n"]
    pc = [c for c in X.columns if c != "subzone_id" and nun[c] > 1]
    g = X.groupby("subzone_id", sort=False)
    S = g[oa].first().join(g[pc].mean().add_suffix("_m"))
    S["nrow"] = g.size()
    S["nrow_tr"] = tr.groupby("subzone_id").size().reindex(S.index).fillna(0).values
    S["sub_n"] = g["sub_n"].first().astype(int)
    S["zone"] = al.groupby("subzone_id", sort=False)["zone_region_id"].first().reindex(S.index).values
    cell = al.groupby(["env_climate_01", "env_climate_03", "env_climate_05"], sort=False).ngroup()
    S["cell"] = cell.groupby(al["subzone_id"]).first().reindex(S.index).values
    S["N"] = household_count(S[HVAC].values)
    S["rest"] = 100 - S[HVAC].sum(axis=1)
    S["K"] = np.round(S.rest * S.N / 100).astype(int)
    for h in HVAC:
        S[h + "_cnt"] = np.round(S[h] * S.N / 100)
    ad = tr.groupby("subzone_id", sort=False)["target_electric_heating_adoption"].first()
    S["ad"] = ad.reindex(S.index).values
    S["cr"] = np.round(S.K - S.ad * S.N / 100)  # NaN for unlabelled subzones
    return S


def neighbour_feats(F, A, size):
    """Pooled count_r / K of fit subzones F around apply subzones A (zone, climate cell, sub_n windows)."""
    out = {}
    for key in ("zone", "cell"):
        sK, sC, n = F.groupby(key).K.sum(), F.groupby(key).cr.sum(), F.groupby(key).size()
        k, c = A[key].map(sK).fillna(0).values, A[key].map(sC).fillna(0).values
        out[f"nb_{key}_q"] = np.where(k > 0, c / np.maximum(k, 1), np.nan)
        out[f"nb_{key}_K"] = k
        out[f"nb_{key}_n"] = A[key].map(n).fillna(0).values
    pos = F.sub_n.values
    cK = np.r_[0, np.cumsum(np.bincount(pos, F.K.values, size))]
    cC = np.r_[0, np.cumsum(np.bincount(pos, F.cr.values, size))]
    cN = np.r_[0, np.cumsum(np.bincount(pos, None, size))]
    for w in WINDOWS:
        lo, hi = np.clip(A.sub_n.values - w, 0, size), np.clip(A.sub_n.values + w + 1, 0, size)
        k, c = cK[hi] - cK[lo], cC[hi] - cC[lo]
        out[f"w{w}_q"] = np.where(k > 0, c / np.maximum(k, 1), np.nan)
        out[f"w{w}_K"] = k
        out[f"w{w}_n"] = cN[hi] - cN[lo]
    return pd.DataFrame(out, index=A.index)


def fit_predict(Fa, La, Fb, Lb, Fs, Ls, mode):
    """Train one model on (Fa, La) with early stopping on (Fb, Lb); return predicted r for Lb and Ls rows."""
    if mode == "q_xent":
        ma, mb = La.K.values > 0, Lb.K.values > 0
        qa, qb = (La.cr / La.K.clip(lower=1)).values, (Lb.cr / Lb.K.clip(lower=1)).values
        m = lgb.train(dict(BASE, objective="cross_entropy"), lgb.Dataset(Fa[ma], qa[ma], weight=La.K.values[ma]),
                      5000, valid_sets=[lgb.Dataset(Fb[mb], qb[mb], weight=Lb.K.values[mb])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        to_r = lambda F, L: 100 * L.K.values * np.clip(m.predict(F, num_iteration=m.best_iteration), 0, 1) / L.N.values  # noqa: E731
    else:
        m = lgb.train(dict(BASE, objective="regression"), lgb.Dataset(Fa, (La.rest - La.ad).values), 5000,
                      valid_sets=[lgb.Dataset(Fb, (Lb.rest - Lb.ad).values)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        to_r = lambda F, L: np.clip(m.predict(F, num_iteration=m.best_iteration), 0, None)  # noqa: E731
    return to_r(Fb, Lb), to_r(Fs, Ls), m.best_iteration


def cv_scores(L, r_oof):
    e2 = (np.clip(L.rest.values - r_oof, 0, 100) - L.ad.values) ** 2
    byn = pd.Series(e2).groupby(np.minimum(L.nrow.values, 6)).mean()
    return np.sqrt(e2.mean()), np.sqrt(sum(UNSEEN_MIX[k] * byn[k] for k in UNSEEN_MIX) / sum(UNSEEN_MIX.values()))


def fit_adoption_v2(cache="cache", verbose=True):
    tr, te = load(cache)
    S = subzone_table(tr, te)
    lab = S.ad.notna().values
    L, U = S[lab], S[~lab]
    size = int(S.sub_n.max()) + 2 * max(WINDOWS) + 2
    feats = [c for c in S.columns if c not in ("zone", "cell", "ad", "cr")]
    modes = ("q_xent", "r_reg")
    oof = {m: np.zeros(len(L)) for m in modes}
    pu = {m: np.zeros(len(U)) for m in modes}
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(L))
    for k, (a, b) in enumerate(folds):
        La, Lb = L.iloc[a], L.iloc[b]
        inner = KFold(5, shuffle=True, random_state=SEED).split(La)
        na = pd.concat([neighbour_feats(La.iloc[ia], La.iloc[ib], size) for ia, ib in inner]).reindex(La.index)
        Fa = La[feats].join(na)
        Fb = Lb[feats].join(neighbour_feats(La, Lb, size))
        Fu = U[feats].join(neighbour_feats(La, U, size))
        for mode in modes:
            rb, ru, it = fit_predict(Fa, La, Fb, Lb, Fu, U, mode)
            oof[mode][b] = rb
            pu[mode] += ru / len(folds)
            if verbose:
                print(f"  adoption fold {k} {mode} it={it}", flush=True)
    r_oof = np.mean([oof[m] for m in modes], axis=0)
    r_u = np.mean([pu[m] for m in modes], axis=0)
    if verbose:
        for m in modes:
            print(f"{m}: CV rmse all {cv_scores(L, oof[m])[0]:.4f} unseen-weighted {cv_scores(L, oof[m])[1]:.4f}")
        s_all, s_un = cv_scores(L, r_oof)
        print(f"average: CV rmse all {s_all:.4f} unseen-weighted {s_un:.4f}", flush=True)
    known = dict(zip(L.index, L.ad.values))
    unseen = dict(zip(U.index, np.clip(U.rest.values - r_u, 0, 100)))
    pred = np.array([known[s] if s in known else unseen[s] for s in te["subzone_id"].values])
    frac_unseen = float(np.mean([s not in known for s in te["subzone_id"].values]))
    return pred, cv_scores(L, r_oof)[1], frac_unseen


if __name__ == "__main__":
    pred, rmse_un, fu = fit_adoption_v2()
    print(f"unseen fraction {fu:.4f}; expected test adoption NRMSE {np.sqrt(fu) * rmse_un / 7.18:.5f}")
    np.save("exp/pred/adopt_v2.npy", pred)
