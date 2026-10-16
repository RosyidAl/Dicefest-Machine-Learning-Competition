"""Phase 12 interpretability: TreeSHAP (LightGBM pred_contrib) per target on 30k validation rows of fold 0.

Models: LightGBM (reference parameters, lr 0.1) on FS_legacy + aggregates + fold-aware target encodings, for
log EUI (t3, and t1 = EUI x area), log elec per meter (t2) and adoption (t5). t4 is an exact identity of bld_dim_02 /
bld_dim_01, reported without a model. Outputs: reports/figures/shap_*.png (300 dpi), reports/shap_family.csv,
reports/shap_top_features.csv.
"""
import os
import sys

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from train_lib import Data  # noqa: E402

FIG = "reports/figures"
os.makedirs(FIG, exist_ok=True)


def family(c):
    for p, f in (("bld_dim_", "bld_dim"), ("bld_era_", "bld_era"), ("hvac_dist_", "hvac_dist"),
                 ("env_climate_", "env_climate"), ("socio_demog_", "socio_demog")):
        if c.startswith(p):
            return f
    if c in ("zone_n", "sub_n"):
        return "ID spasial"
    if c.startswith(("sz_", "zn_", "w2_", "w10_", "c0x_")):
        return "target encoding spasial"
    if c.startswith(("sub_", "zon_")):
        return "agregat grup"
    return "rasio fisika"


d = Data(n_folds=5)
ex, ext = d.agg_features()
a, b = d.folds[0]
X = pd.concat([d.X.reset_index(drop=True), ex], axis=1)
Xa = pd.concat([X.iloc[a].reset_index(drop=True), d.te_features(a, "oof", (2, 10), True)], axis=1)
Xb = pd.concat([X.iloc[b].reset_index(drop=True), d.te_features(a, d.te_apply(b), (2, 10), True)], axis=1)
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(b), 30000, replace=False))
y = d.y
M = d.M.astype(float)
TGT = {"t3 EUI (log)": np.log(y["target_eui_gas_kwh_per_sqm"]),
       "t2 elec per meter (log)": np.log(y["target_macro_elec_kwh"] / M),
       "t5 adopsi": y["target_electric_heating_adoption"]}
P = dict(objective="regression", learning_rate=0.1, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, deterministic=True, force_row_wise=True,
         num_threads=8, verbose=-1)
fam_rows, top_rows = [], []
for name, t in TGT.items():
    m = lgb.train(P, lgb.Dataset(Xa, t[a]), 5000, valid_sets=[lgb.Dataset(Xb, t[b])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    S = m.predict(Xb.iloc[sel], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(S).mean(0), Xb.columns)
    fam = imp.groupby([family(c) for c in imp.index]).sum().sort_values(ascending=False)
    for f, v in fam.items():
        fam_rows.append({"target": name, "family": f, "mean_abs_shap": v, "share": v / fam.sum()})
    for c, v in imp.sort_values(ascending=False).head(15).items():
        top_rows.append({"target": name, "feature": c, "family": family(c), "mean_abs_shap": v})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    (fam / fam.sum()).iloc[::-1].plot.barh(ax=ax[0], color="C0")
    ax[0].set_title(f"{name}: porsi |SHAP| per keluarga fitur")
    imp.sort_values().tail(15).plot.barh(ax=ax[1], color="C1")
    ax[1].set_title(f"{name}: 15 fitur teratas (mean |SHAP|)")
    plt.tight_layout()
    tag = name.split()[0]
    plt.savefig(f"{FIG}/shap_{tag}.png", dpi=300)
    plt.close()
    top3 = imp.sort_values(ascending=False).index[:3]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, c in enumerate(top3):
        j = list(Xb.columns).index(c)
        ax[k].scatter(Xb.iloc[sel][c], S[:, j], s=1, alpha=0.3)
        ax[k].set_xlabel(c)
        ax[k].set_ylabel("SHAP")
    plt.suptitle(f"{name}: dependence plot fitur utama")
    plt.tight_layout()
    plt.savefig(f"{FIG}/shap_dep_{tag}.png", dpi=300)
    plt.close()
    print(name, "best_iter", m.best_iteration, fam.round(4).to_dict(), flush=True)
pd.DataFrame(fam_rows).to_csv("reports/shap_family.csv", index=False)
pd.DataFrame(top_rows).to_csv("reports/shap_top_features.csv", index=False)
print("done")
