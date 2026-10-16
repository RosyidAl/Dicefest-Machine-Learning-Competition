import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, "src")
from common import *

tr, te = load()
const = constant_cols(tr)
X = base_features(tr, const)
X["subzone_id"] = tr["subzone_id"].values
nun = X.groupby("subzone_id").nunique().max()
oa_cols = [c for c in X.columns if c != "subzone_id" and nun[c] <= 1]
pc_cols = [c for c in X.columns if c != "subzone_id" and nun[c] > 1 and c not in ("sub_n",)]
print("OA-level cols", len(oa_cols), "postcode-level cols", len(pc_cols))
S = X.groupby("subzone_id", sort=False)[oa_cols].first()
S = S.join(X.groupby("subzone_id", sort=False)[pc_cols].mean().add_suffix("_m"))
S["n_rows"] = X.groupby("subzone_id", sort=False).size()
S["sub_n"] = X.groupby("subzone_id", sort=False)["sub_n"].first()
y_ad = tr.groupby("subzone_id", sort=False)["target_electric_heating_adoption"].first().reindex(S.index).values
rest = S["hvac_rest"].values
r = rest - y_ad
kf = KFold(5, shuffle=True, random_state=SEED)
p = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED, deterministic=True, force_row_wise=True,
         num_threads=12, verbose=-1)

def cv(Xs, target, name):
    oof = np.zeros(len(Xs)); its = []
    for a, b in kf.split(Xs):
        d = lgb.Dataset(Xs.iloc[a], target[a]); v = lgb.Dataset(Xs.iloc[b], target[b])
        m = lgb.train(p, d, 3000, valid_sets=[v], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(Xs.iloc[b], num_iteration=m.best_iteration); its.append(m.best_iteration)
    print(name, its)
    return oof

t = time.time()
base = rest - r.mean()
print("naive rest-mean(r): rmse", np.sqrt(np.mean((np.clip(base, 0, None) - y_ad) ** 2)))
o = cv(S, r, "r-model")
pred = np.clip(rest - np.clip(o, 0, None), 0, None)
print("r-model rmse", np.sqrt(np.mean((pred - y_ad) ** 2)), time.time() - t)
np.save("oof/adopt_r_sub.npy", o)
o2 = cv(S, y_ad, "direct")
print("direct rmse", np.sqrt(np.mean((np.clip(o2, 0, None) - y_ad) ** 2)))
