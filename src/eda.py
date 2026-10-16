"""Phase 1 EDA and data forensics -> reports/eda/findings.json + reports/eda/figures/*.png.

Sections follow MASTER_PROMPT 5.A-5.E. Heavy steps use subsamples where noted. Run: python src/eda.py
"""
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import load  # noqa: E402
from metric import OFFICIAL_SIGMAS, TARGET_COLS  # noqa: E402

FIG = "reports/eda/figures"
os.makedirs(FIG, exist_ok=True)
PREF = ["bld_dim_", "bld_era_", "hvac_dist_", "env_climate_", "socio_demog_"]
T1, T2, T3, T4, T5 = TARGET_COLS
F = {}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{name}.png", dpi=150)
    plt.close()


tr, te = load()
feat = sorted([c for c in tr.columns if c.startswith(tuple(PREF))])
num_feat = [c for c in feat if pd.api.types.is_numeric_dtype(tr[c])]
str_feat = [c for c in feat if c not in num_feat]

# ---------------- 5.A integrity ----------------
A = {}
A["string_features"] = {c: tr[c].value_counts().head(6).to_dict() for c in str_feat}
A["nan_train"] = {c: int(v) for c, v in tr[feat].isna().sum().items() if v}
A["nan_test"] = {c: int(v) for c, v in te[feat].isna().sum().items() if v}
A["inf_train"] = int(np.isinf(tr[num_feat].values).sum())
A["constant_cols"] = [c for c in feat if tr[c].nunique(dropna=False) <= 1]
sent = {}
for c in num_feat:
    v = tr[c].values
    for s in (-999, -1, 9999, 99999):
        k = int((v == s).sum())
        if k:
            sent.setdefault(c, {})[str(s)] = k
A["sentinel_values"] = sent
A["zero_share_gt_50pct"] = {c: round(float((tr[c] == 0).mean()), 3) for c in num_feat if (tr[c] == 0).mean() > 0.5}
h = pd.util.hash_pandas_object(tr[num_feat].T, index=False)
dup_cols = h[h.duplicated(keep=False)]
A["exact_duplicate_columns"] = [list(g.index) for _, g in dup_cols.groupby(dup_cols)] if len(dup_cols) else []
sub = tr[num_feat].sample(50000, random_state=0)
cm = np.corrcoef(np.nan_to_num(sub.values.astype(float)), rowvar=False)
iu = np.triu_indices_from(cm, 1)
hi = [(num_feat[i], num_feat[j], round(float(cm[i, j]), 5)) for i, j in zip(*iu) if abs(cm[i, j]) > 0.999]
A["pairs_abs_r_gt_0999"] = hi
A["n_pairs_abs_r_gt_0999"] = len(hi)
key = feat + ["subzone_id", "zone_region_id"]
htr = pd.util.hash_pandas_object(tr[key], index=False)
hte = pd.util.hash_pandas_object(te[key], index=False)
A["dup_rows_train"] = int(htr.duplicated().sum())
A["dup_rows_test"] = int(hte.duplicated().sum())
A["test_rows_identical_to_train"] = int(hte.isin(set(htr)).sum())
g = tr.assign(h=htr.values).groupby("h")
multi = g.filter(lambda x: len(x) > 1) if A["dup_rows_train"] else tr.iloc[:0]
if len(multi):
    A["train_dup_groups_target_identical"] = bool((multi.groupby(htr[multi.index].values)[TARGET_COLS].nunique() == 1)
                                                  .all().all())
A["integer_share"] = {c: round(float(np.mean(np.mod(tr[c].dropna().values, 1) == 0)), 3) for c in num_feat}
A["n_integer_cols"] = int(sum(v == 1.0 for v in A["integer_share"].values()))
F["A_integrity"] = A
log("5.A done")

# ---------------- 5.B split forensics ----------------
B = {}
idn_tr = tr["id"].str[3:].astype(int).values
idn_te = te["id"].str[3:].astype(int).values
B["id_train_range"] = [int(idn_tr.min()), int(idn_tr.max())]
B["id_test_range"] = [int(idn_te.min()), int(idn_te.max())]
B["id_train_contiguous"] = bool(np.array_equal(np.sort(idn_tr), np.arange(idn_tr.min(), idn_tr.max() + 1)))
B["n_ratio_train_total"] = round(len(tr) / (len(tr) + len(te)), 6)
zn = tr["zone_region_id"].str[4:].astype(int)
B["spearman_id_vs"] = {c: round(float(stats.spearmanr(idn_tr, tr[c])[0]), 4) for c in TARGET_COLS}
B["spearman_id_vs"]["zone_n"] = round(float(stats.spearmanr(idn_tr, zn)[0]), 4)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
o = np.argsort(idn_tr)
for k, c in enumerate([T1, T2, T3]):
    ax[k].plot(pd.Series(tr[c].values[o]).rolling(5000).mean().values)
    ax[k].set_title(f"rolling mean {c} vs id")
savefig("id_rolling_mean")
for lvl in ("zone_region_id", "subzone_id"):
    s_tr, s_te = set(tr[lvl]), set(te[lvl])
    cnt_tr, cnt_te = tr[lvl].value_counts(), te[lvl].value_counts()
    both = pd.concat([cnt_tr, cnt_te], axis=1, keys=["tr", "te"]).fillna(0)
    ratio = both["te"] / (both["tr"] + both["te"])
    d = {"n_train": len(s_tr), "n_test": len(s_te), "n_common": len(s_tr & s_te),
         "test_rows_with_group_in_train": round(float(te[lvl].isin(s_tr).mean()), 4),
         "rows_per_group_quantiles": both.sum(axis=1).quantile([0, .1, .5, .9, 1]).round(1).tolist(),
         "test_ratio_mean": round(float(ratio.mean()), 4)}
    if lvl == "zone_region_id":
        n = both["tr"] + both["te"]
        z = (both["te"] - 0.3 * n) / np.sqrt(0.3 * 0.7 * n)
        d["test_ratio_z_std"] = round(float(z.std()), 3)  # ~1 under a random row split
        d["test_ratio_range"] = [round(float(ratio.min()), 3), round(float(ratio.max()), 3)]
        plt.figure(figsize=(6, 3.5))
        plt.hist(ratio, bins=40)
        plt.axvline(0.3, color="r")
        plt.title("zone: n_test / (n_train + n_test)")
        savefig("zone_test_ratio")
    else:
        single = both[(both.sum(axis=1) == 1)]
        d["groups_with_one_row"] = int(len(single))
        # expected share of test rows in unseen subzones under a random 30% row split
        n = (both["tr"] + both["te"]).values
        d["expected_unseen_test_share_random_split"] = round(float((n * 0.3 * 0.3 ** (n - 1)).sum() / (0.3 * n.sum())), 4)
        d["observed_unseen_test_share"] = round(1 - d["test_rows_with_group_in_train"], 4)
    B[lvl] = d
allr = pd.concat([tr[["zone_region_id", "subzone_id"]], te[["zone_region_id", "subzone_id"]]])
B["subzone_in_one_zone_only"] = bool((allr.groupby("subzone_id")["zone_region_id"].nunique() == 1).all())
sn = allr["subzone_id"].str[4:].astype(int).values
zz = allr["zone_region_id"].str[4:].astype(int).values
oo = np.argsort(sn)
B["consecutive_subzone_same_zone_share"] = round(float(np.mean(zz[oo][1:] == zz[oo][:-1])), 4)
B["subzone_id_examples"] = allr["subzone_id"].head(3).tolist()
log("5.B groups done")


def adversarial(cols, tag):
    X = pd.concat([tr[cols], te[cols]], ignore_index=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    oof = np.zeros(len(y))
    imp = np.zeros(X.shape[1])
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, subsample=0.5, subsample_freq=1,
                               colsample_bytree=0.5, num_threads=8, seed=42, deterministic=True, force_row_wise=True,
                               verbose=-1).fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        imp += m.booster_.feature_importance("gain")
    top = pd.Series(imp, X.columns).sort_values(ascending=False).head(10)
    return {"auc": round(float(roc_auc_score(y, oof)), 4), "top_gain_features": [str(c) for c in top.index]}


B["adversarial_no_ids"] = adversarial(feat, "noid")
B["adversarial_with_ids"] = adversarial(feat + ["zone_region_id", "subzone_id"], "id")
F["B_split"] = B
log(f"5.B adversarial done {B['adversarial_no_ids']['auc']} {B['adversarial_with_ids']['auc']}")

# ---------------- 5.C targets ----------------
C = {}
for c in TARGET_COLS:
    v = tr[c].values
    sq = (v - v.mean()) ** 2
    top1 = np.sort(sq)[::-1][: len(v) // 100].sum() / sq.sum()
    C[c] = {"mean": float(v.mean()), "std_ddof0": float(v.std()), "std_ddof1": float(v.std(ddof=1)),
            "sigma_frozen": OFFICIAL_SIGMAS[c], "std_over_sigma": round(float(v.std() / OFFICIAL_SIGMAS[c]), 4),
            "min": float(v.min()), "max": float(v.max()),
            "quantiles": dict(zip(["0.1%", "1%", "50%", "99%", "99.9%"],
                                  np.quantile(v, [.001, .01, .5, .99, .999]).round(4).tolist())),
            "skew": round(float(stats.skew(v)), 3), "n_zero": int((v == 0).sum()),
            "top1pct_share_of_sq_dev": round(float(top1), 4)}
    if c in (T4, T5):
        C[c]["out_of_0_100"] = int(((v < 0) | (v > 100)).sum())
C["pearson"] = tr[TARGET_COLS].corr().round(3).values.tolist()
C["spearman"] = tr[TARGET_COLS].corr("spearman").round(3).values.tolist()
fig, ax = plt.subplots(2, 5, figsize=(20, 6))
for k, c in enumerate(TARGET_COLS):
    ax[0, k].hist(tr[c], bins=100)
    ax[0, k].set_title(c.replace("target_", ""), fontsize=9)
    ax[1, k].hist(np.log1p(tr[c]), bins=100, color="C1")
    ax[1, k].set_title("log1p", fontsize=9)
savefig("target_hist")
# OOF group-mean predictors (5-fold random) -> NRMSE: size of the group-level signal
gm = {}
for lvl in ("zone_region_id", "subzone_id"):
    gm[lvl] = {}
    for c in TARGET_COLS:
        p = np.zeros(len(tr))
        for a, b in KFold(5, shuffle=True, random_state=42).split(tr):
            means = tr.iloc[a].groupby(lvl)[c].mean()
            p[b] = tr[lvl].iloc[b].map(means).fillna(tr[c].iloc[a].mean()).values
        gm[lvl][c] = round(float(np.sqrt(np.mean((p - tr[c].values) ** 2)) / OFFICIAL_SIGMAS[c]), 4)
C["oof_group_mean_nrmse"] = gm
F["C_targets"] = C
log("5.C done")

# ---------------- 5.D features ----------------
D = {}
D["prefix_stats"] = {}
for p in PREF:
    cols = [c for c in num_feat if c.startswith(p)]
    vals = tr[cols]
    rs = vals.sum(axis=1)
    D["prefix_stats"][p] = {"n_cols": len(cols), "n_cols_numeric": len(cols), "min": float(vals.min().min()),
                            "max": float(vals.max().max()), "n_cols_with_negative": int((vals.min() < 0).sum()),
                            "row_sum_quantiles": rs.quantile([0, .01, .5, .99, 1]).round(3).tolist()}
# compositions: contiguous blocks whose row sum is ~constant
comp = []
for p in PREF:
    cols = sorted([c for c in num_feat if c.startswith(p)])
    for i in range(len(cols)):
        for j in range(i + 2, min(i + 16, len(cols)) + 1):
            s = tr[cols[i:j]].sum(axis=1)
            if s.std() < 0.01 * max(abs(s.mean()), 1e-9) and abs(s.mean()) > 0:
                comp.append({"cols": f"{cols[i]}..{cols[j - 1]}", "n": j - i, "row_sum_mean": round(float(s.mean()), 4),
                             "row_sum_std": round(float(s.std()), 5)})
D["constant_sum_blocks"] = comp
hv = tr[[f"hvac_dist_{i:02d}" for i in range(1, 11)]].sum(axis=1)
D["hvac_row_sum_quantiles"] = hv.quantile([0, .01, .5, .99, 1]).round(3).tolist()
era = tr[[f"bld_era_{i:02d}" for i in range(1, 16)]]
D["era_row_sum_quantiles"] = era.sum(axis=1).quantile([0, .01, .5, .99, 1]).round(3).tolist()
# within-region variance ratio
wr = {}
for c in num_feat:
    v = tr[c].astype(float)
    tot = v.var()
    if tot > 0:
        wr[c] = round(float((v - v.groupby(tr["zone_region_id"]).transform("mean")).var() / tot), 4)
D["within_zone_var_ratio_lowest"] = dict(sorted(wr.items(), key=lambda x: x[1])[:15])
D["n_features_within_zone_ratio_lt_0.05"] = int(sum(v < 0.05 for v in wr.values()))
wsub = {}
for c in [c for c in num_feat if c.startswith(("socio_demog_", "hvac_dist_", "env_climate_"))]:
    v = tr[c].astype(float)
    if v.var() > 0:
        wsub[c] = float((v - v.groupby(tr["subzone_id"]).transform("mean")).var() / v.var())
D["n_census_cols_constant_within_subzone"] = int(sum(v < 1e-12 for v in wsub.values()))
D["n_census_cols_checked"] = len(wsub)
# redundancy clusters (Spearman on subsample)
sp = sub.rank().corr().values
D["n_pairs_abs_spearman_gt_0.95"] = int((np.abs(sp[iu]) > 0.95).sum())
# univariate relations
uni = {}
s2 = tr.sample(60000, random_state=1)
for c in TARGET_COLS:
    r = {f: abs(stats.spearmanr(s2[f], s2[c])[0]) for f in num_feat if s2[f].nunique() > 1}
    uni[c] = {k: round(float(v), 3) for k, v in sorted(r.items(), key=lambda x: -x[1])[:8]}
D["top_abs_spearman"] = uni
imp = {}
Xs = s2[num_feat]
for c in TARGET_COLS:
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=63, num_threads=8, seed=42,
                          deterministic=True, force_row_wise=True, verbose=-1).fit(Xs, s2[c])
    gi = pd.Series(m.booster_.feature_importance("gain"), num_feat)
    imp[c] = (gi / gi.sum()).sort_values(ascending=False).head(8).round(3).to_dict()
D["lgb_gain_top"] = imp
F["D_features"] = D
log("5.D done")

# ---------------- 5.E physics ----------------
E = {}
pos = [c for c in num_feat if c.startswith("bld_dim_") and (tr[c] > 0).all()]
stab = {}
for c in pos:
    r = tr[T1] / (tr[T3] * tr[c])
    ok = np.isfinite(r) & (tr[T3] > 0)
    stab[c] = round(float(np.std(np.log(r[ok]))), 6)
E["gas_over_eui_times_f_logstd"] = dict(sorted(stab.items(), key=lambda x: x[1])[:5])
best = min(stab, key=stab.get)
E["floor_area_col"] = best
E["max_abs_rel_err_gas_eq_eui_x_area"] = float(np.max(np.abs(tr[T3] * tr[best] / tr[T1] - 1)))
st2 = {}
for c in pos:
    st2[c] = round(float(np.std(np.log(tr[T2] / tr[c]))), 4)
E["elec_over_f_logstd_best"] = dict(sorted(st2.items(), key=lambda x: x[1])[:5])
clean = (100 * tr["bld_dim_02"] / tr["bld_dim_01"]).round(2)
E["clean_identity_exact_share"] = float(np.mean(np.isclose(clean, tr[T4], atol=1e-9)))
E["adoption_constant_within_subzone"] = bool((tr.groupby("subzone_id")[T5].nunique() == 1).all())
hcols = [f"hvac_dist_{i:02d}" for i in range(1, 11)]
from sklearn.linear_model import LinearRegression  # noqa: E402

lr5 = LinearRegression().fit(tr[hcols], tr[T5])
E["adoption_r2_linear_hvac"] = round(float(lr5.score(tr[hcols], tr[T5])), 4)
rest = 100 - tr[hcols].sum(axis=1)
E["adoption_le_hvac_rest_share"] = float(np.mean(tr[T5] <= rest + 1e-6))
ecols = [f"bld_era_{i:02d}" for i in range(1, 16)]
E["spearman_era_share_vs_eui"] = {c: round(float(stats.spearmanr(tr[c], tr[T3])[0]), 3) for c in ecols}
lr4 = LinearRegression().fit(tr[ecols + hcols], tr[T4])
E["clean_r2_linear_era_hvac"] = round(float(lr4.score(tr[ecols + hcols], tr[T4])), 4)
plt.figure(figsize=(6, 4))
plt.scatter(np.log(tr[T3] * tr[best]).sample(20000, random_state=0),
            np.log(tr[T1]).sample(20000, random_state=0), s=1)
plt.xlabel(f"log(EUI x {best})")
plt.ylabel("log(gas)")
plt.title("gas = EUI x floor area")
savefig("physics_gas_eui_area")
F["E_physics"] = E
log("5.E done")

json.dump(F, open("reports/eda/findings.json", "w"), indent=1, default=str)
log("wrote reports/eda/findings.json")
