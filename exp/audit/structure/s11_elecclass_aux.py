"""Auxiliary-target test: the elec 'multi-decimal' class (elec not a multiple of 0.1; share rises with M, plausibly
Economy-7 meters) is a target property. Fit P(class | features) with LightGBM on P rows only, predict H, then check
(5-fold cross-fitted on H) whether the class probability explains the honest residuals of gas / EUI / elec."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src")
from common import base_features, constant_cols
S = {"gas": 151750.04, "elec": 41725.37, "eui": 40.65}
tr = pd.read_parquet("cache/train.parquet")
e = tr.target_macro_elec_kwh.values
cls = (np.abs(e * 10 - np.round(e * 10)) > 1e-4).astype(int)
X = base_features(tr, constant_cols(tr)); X = X[sorted(X.columns)]
del tr
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); P, H = d["P"], d["H"]
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, seed=0, num_threads=2, deterministic=True, force_row_wise=True, verbose=-1)
# small internal split of P for early stopping
rng = np.random.default_rng(0); m = rng.random(len(P)) < 0.9
dtr = lgb.Dataset(X.values[P[m]], cls[P[m]]); dva = lgb.Dataset(X.values[P[~m]], cls[P[~m]])
b = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
p = b.predict(X.values[H], num_iteration=b.best_iteration)
print("best iter", b.best_iteration, "AUC on H", round(roc_auc_score(cls[H], p), 4), "base rate", cls.mean().round(4))
np.save("exp/audit/structure/s11_pclass_H.npy", p)
imp = pd.Series(b.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("top features", imp.head(8).round(0).to_dict())
M = d["M"][H]
from numpy.linalg import lstsq
for mt in ["gas", "eui", "elec"]:
    y = d[f"{mt}_y"]; pc = d[f"{mt}_corrected"]; r = y - pc
    Z = np.column_stack([p - p.mean(), (p - p.mean()) * pc])
    folds = np.random.default_rng(1).integers(0, 5, len(H))
    adj = np.zeros(len(H))
    for f in range(5):
        tr_, te_ = folds != f, folds == f
        w = lstsq(Z[tr_], r[tr_], rcond=None)[0]; adj[te_] = Z[te_] @ w
    base = np.sqrt(np.mean(r ** 2)) / S[mt]; new = np.sqrt(np.mean((r - adj) ** 2)) / S[mt]
    print(f"{mt}: H NRMSE {base:.5f} -> {new:.5f}  delta {new - base:+.6f}  (total {(new - base) / 5:+.6f})")
