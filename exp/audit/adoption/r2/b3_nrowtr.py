"""Adoption lens (round 2), check b3: train/test feature shift of `nrow_tr` in the adopt_v2 unseen-subzone model.

adopt_v2 uses `nrow_tr` (number of the subzone's rows in train) as a feature. Every labelled subzone has
nrow_tr >= 1, every unseen test subzone has nrow_tr = 0 (LightGBM bins 0 with 1). The 5-fold subzone CV therefore
never scores the condition the test is in. Given nrow, nrow_tr is pure split noise (pooled q is flat in nrow_tr), but
it is a proxy for nrow in trees that do not sample nrow (feature_fraction 0.5), and q depends strongly on nrow.

Honest A/B (same folds, same LightGBM settings, code of src/adopt_v2.py):
  A  production features (replica; must reproduce exp/adopt/base_*_oof.npy and the unseen part of
     exp/pred/adopt_v2.npy byte for byte)
     A0: validation subzones scored with their true nrow_tr (the CV number adopt_v2 reports, 1.0303)
     A1: validation subzones scored with nrow_tr := 0  (what an unseen test subzone looks like)
  B  same without the nrow_tr feature (validation and test look alike)
Score: RMSE of adoption on labelled subzones, weighted to the unseen test row mix (adopt_v2.cv_scores).
The table is built memory-lean (base_features per frame, no full train+test concat), max RSS ~2 GB.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/adoption/r2/b3_nrowtr.py
"""
import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from adopt_v2 import BASE, WINDOWS, cv_scores, household_count, neighbour_feats  # noqa: E402
from common import HVAC, SEED, base_features, constant_cols  # noqa: E402

AD = "target_electric_heating_adoption"
OUT = "exp/audit/adoption/r2"
SMALL = ["subzone_id", "zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"]
t0 = time.time()


def lean_subzone_table():
    """identical to adopt_v2.subzone_table(tr, te) but without holding the full train + test frames at once"""
    tr = pd.read_parquet("cache/train.parquet")
    const = constant_cols(tr)
    Xtr = base_features(tr, const)
    small_tr = tr[SMALL + [AD]].copy()
    del tr
    gc.collect()
    te = pd.read_parquet("cache/test.parquet")
    Xte = base_features(te, const)
    small_te = te[SMALL].copy()
    del te
    gc.collect()
    ntr = len(Xtr)
    X = pd.concat([Xtr, Xte], ignore_index=True)
    del Xtr, Xte
    gc.collect()
    al = pd.concat([small_tr[SMALL], small_te], ignore_index=True)
    X["subzone_id"] = al["subzone_id"].values
    nun = X.iloc[:ntr].groupby("subzone_id").nunique().max()
    oa = [c for c in X.columns if c != "subzone_id" and nun[c] <= 1 and c != "sub_n"]
    pc = [c for c in X.columns if c != "subzone_id" and nun[c] > 1]
    g = X.groupby("subzone_id", sort=False)
    S = g[oa].first().join(g[pc].mean().add_suffix("_m"))
    S["nrow"] = g.size()
    S["nrow_tr"] = small_tr.groupby("subzone_id").size().reindex(S.index).fillna(0).values
    S["sub_n"] = g["sub_n"].first().astype(int)
    S["zone"] = al.groupby("subzone_id", sort=False)["zone_region_id"].first().reindex(S.index).values
    cell = al.groupby(["env_climate_01", "env_climate_03", "env_climate_05"], sort=False).ngroup()
    S["cell"] = cell.groupby(al["subzone_id"]).first().reindex(S.index).values
    S["N"] = household_count(S[HVAC].values)
    S["rest"] = 100 - S[HVAC].sum(axis=1)
    S["K"] = np.round(S.rest * S.N / 100).astype(int)
    for h in HVAC:
        S[h + "_cnt"] = np.round(S[h] * S.N / 100)
    ad = small_tr.groupby("subzone_id", sort=False)[AD].first()
    S["ad"] = ad.reindex(S.index).values
    S["cr"] = np.round(S.K - S.ad * S.N / 100)
    te_sz = small_te["subzone_id"].values
    del X, g
    gc.collect()
    return S, te_sz


def fit(Fa, La, Fb, Lb, mode, apply_sets):
    """adopt_v2.fit_predict, returning r predictions for several (F, L) sets"""
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
    imp = dict(zip(m.feature_name(), m.feature_importance("gain")))
    return [to_r(F, L) for F, L in apply_sets], m.best_iteration, imp


S, te_sz = lean_subzone_table()
print(f"table {S.shape} {time.time() - t0:.0f}s", flush=True)
if "--save-table" in sys.argv:
    S.to_parquet(f"{OUT}/S_v2.parquet")
SMOKE = "--smoke" in sys.argv
ARMS = [a for a in ("A", "B") if a in (sys.argv[sys.argv.index("--arms") + 1] if "--arms" in sys.argv else "AB")]
if SMOKE:
    S = S.sample(20000, random_state=SEED)
lab = S.ad.notna().values
L, U = S[lab], S[~lab]
size = int(S.sub_n.max()) + 2 * max(WINDOWS) + 2
feats_A = [c for c in S.columns if c not in ("zone", "cell", "ad", "cr")]
feats_B = [c for c in feats_A if c != "nrow_tr"]
folds = list(KFold(5, shuffle=True, random_state=SEED).split(L))
modes = ("q_xent", "r_reg")
oof = {k: {m: np.zeros(len(L)) for m in modes} for k in ("A0", "A1", "B")}
pu = {k: {m: np.zeros(len(U)) for m in modes} for k in ("A", "B")}
imp_nrow_tr = {m: [] for m in modes}
for k, (a, b) in enumerate(folds):
    La, Lb = L.iloc[a], L.iloc[b]
    inner = KFold(5, shuffle=True, random_state=SEED).split(La)
    na = pd.concat([neighbour_feats(La.iloc[ia], La.iloc[ib], size) for ia, ib in inner]).reindex(La.index)
    nb, nu = neighbour_feats(La, Lb, size), neighbour_feats(La, U, size)
    for arm, feats in [(x, {"A": feats_A, "B": feats_B}[x]) for x in ARMS]:
        Fa, Fb, Fu = La[feats].join(na), Lb[feats].join(nb), U[feats].join(nu)
        sets = [(Fb, Lb), (Fu, U)]
        if arm == "A":
            Fb0 = Fb.copy()
            Fb0["nrow_tr"] = 0.0
            sets.append((Fb0, Lb))
        for mode in modes:
            preds, it, imp = fit(Fa, La, Fb, Lb, mode, sets)
            if arm == "A":
                oof["A0"][mode][b], oof["A1"][mode][b] = preds[0], preds[2]
                tot = sum(imp.values())
                imp_nrow_tr[mode].append(imp["nrow_tr"] / tot)
            else:
                oof["B"][mode][b] = preds[0]
            pu[arm][mode] += preds[1] / len(folds)
            print(f"fold {k} arm {arm} {mode} it={it} {time.time() - t0:.0f}s", flush=True)

res = {}
# replication check against the saved production arrays
for m in modes if not SMOKE else ():
    ref = np.load(f"exp/adopt/base_{m}_oof.npy")
    res[f"replica_oof_{m}_max_absdiff"] = float(np.abs(ref - oof["A0"][m]).max())
    refu = np.load(f"exp/adopt/base_{m}_unseen.npy")
    res[f"replica_unseen_{m}_max_absdiff"] = float(np.abs(refu - pu["A"][m]).max())
r_uA = np.mean([pu["A"][m] for m in modes], axis=0)
r_uB = np.mean([pu["B"][m] for m in modes], axis=0) if "B" in ARMS else r_uA
known = dict(zip(L.index, L.ad.values))
if SMOKE:
    te_sz = np.array([s for s in te_sz if s in S.index])
predA = np.array([known.get(s, np.nan) for s in te_sz])
predB = predA.copy()
umap_A = dict(zip(U.index, np.clip(U.rest.values - r_uA, 0, 100)))
umap_B = dict(zip(U.index, np.clip(U.rest.values - r_uB, 0, 100)))
unseen_rows = np.isnan(predA)
predA[unseen_rows] = [umap_A[s] for s in te_sz[unseen_rows]]
predB[unseen_rows] = [umap_B[s] for s in te_sz[unseen_rows]]
if not SMOKE:
    prod = np.load("exp/pred/adopt_v2.npy")
    res["replica_test_vs_adopt_v2_max_absdiff"] = float(np.abs(predA - prod).max())
    if "B" in ARMS:
        np.save(f"{OUT}/b3_adopt_noNrowTr.npy", predB)
        np.save(f"{OUT}/b3_rB_unseen.npy", r_uB)
KEYS = ("A0", "A1", "B") if "B" in ARMS else ("A0", "A1")
for key in KEYS:
    for m in modes:
        res[f"{key}_{m}_unseen_weighted"] = round(cv_scores(L, oof[key][m])[1], 5)
    r_hat = np.mean([oof[key][m] for m in modes], axis=0)
    res[f"{key}_avg_all"], res[f"{key}_avg_unseen_weighted"] = (round(x, 5) for x in cv_scores(L, r_hat))
    if not SMOKE:
        np.save(f"{OUT}/b3_oof_{key}.npy", r_hat)
# per nrow bucket MSE
for key in KEYS:
    r_hat = np.mean([oof[key][m] for m in modes], axis=0)
    e2 = (np.clip(L.rest.values - r_hat, 0, 100) - L.ad.values) ** 2
    res[f"{key}_mse_by_nrow"] = {int(i): round(float(v), 4) for i, v in
                                 pd.Series(e2).groupby(np.minimum(L.nrow.values, 6)).mean().items()}
    res[f"{key}_mean_r_by_nrow"] = {int(i): round(float(v), 4) for i, v in
                                    pd.Series(r_hat).groupby(np.minimum(L.nrow.values, 6)).mean().items()}
res["true_mean_r_by_nrow"] = {int(i): round(float(v), 4) for i, v in
                              pd.Series(L.rest.values - L.ad.values).groupby(np.minimum(L.nrow.values, 6)).mean().items()}
res["nrow_tr_gain_share"] = {m: [round(x, 4) for x in v] for m, v in imp_nrow_tr.items()}
d = r_uB - r_uA
res["unseen_r_shift_B_minus_A_by_nrow"] = {int(i): round(float(v), 4) for i, v in
                                          pd.Series(d).groupby(np.minimum(U.nrow.values, 6)).mean().items()}
res["unseen_rms_diff_B_vs_A_rows"] = float(np.sqrt(np.mean((predB - predA)[unseen_rows] ** 2)))
json.dump(res, open(f"{OUT}/b3_nrowtr{'_smoke' if SMOKE else ''}.json", "w"), indent=1)
for k, v in res.items():
    print(k, v)
print(f"done {time.time() - t0:.0f}s")
