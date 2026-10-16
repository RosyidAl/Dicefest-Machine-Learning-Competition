import sys, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold
sys.path.insert(0, "src")
from common import *
from te import *

tr, te_ = load()
const = constant_cols(tr)
X0 = base_features(tr, const)
A = tr["bld_dim_06"].values; M = tr["bld_dim_12"].values
y_gas, y_elec, y_eui = [tr[c].values for c in TARGET_COLS[:3]]
sub = tr["subzone_id"].values; zone = tr["zone_region_id"].values
# values to encode
V = np.c_[y_eui, np.log(y_eui), y_elec / M, np.log(y_elec / M), y_elec / A, y_gas / M, np.log(y_gas)]
VN = ["eui", "leui", "epm", "lepm", "epa", "gpm", "lgas"]
# neighbour feature means (to judge similarity)
W = np.c_[np.log(A), np.log(M)]
WN = ["lA", "lM"]

def te_block(fit_idx, apply_idx=None):
    """TE features: OOF inside fit_idx if apply_idx is None, else fit->apply."""
    vals = np.c_[V, W]
    if apply_idx is None:
        s, c = oof_group_stats(sub[fit_idx], vals[fit_idx]); sz, cz = oof_group_stats(zone[fit_idx], V[fit_idx])
    else:
        s, c = group_stats(sub[fit_idx], vals[fit_idx], sub[apply_idx]); sz, cz = group_stats(zone[fit_idx], V[fit_idx], zone[apply_idx])
    f = stats_to_features(s, c, VN + WN, "sz")
    fz = stats_to_features(sz, cz, VN, "zn")
    return pd.concat([f, fz], axis=1)

kf = KFold(5, shuffle=True, random_state=SEED)
params = dict(n_estimators=6000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6,
              min_child_weight=5, reg_lambda=1.0, device="cuda", tree_method="hist", random_state=SEED,
              early_stopping_rounds=200, eval_metric="rmse")
oof = {k: np.zeros(len(tr)) for k in ["eui", "elec"]}
t = time.time()
for f, (a, b) in enumerate(kf.split(X0)):
    Ta = pd.concat([X0.iloc[a].reset_index(drop=True), te_block(a)], axis=1)
    Tb = pd.concat([X0.iloc[b].reset_index(drop=True), te_block(a, b)], axis=1)
    for k, yv, w, mult in [("eui", y_eui, None, 1.0), ("elec", y_elec / M, M ** 2 / np.mean(M ** 2), M)]:
        m = xgb.XGBRegressor(**params)
        m.fit(Ta, yv[a], sample_weight=None if w is None else w[a], eval_set=[(Tb, yv[b])],
              sample_weight_eval_set=None if w is None else [w[b]], verbose=False)
        oof[k][b] = m.predict(Tb) * (1.0 if np.isscalar(mult) else mult[b])
        print(f"fold {f} {k} it {m.best_iteration} {time.time()-t:.0f}s", flush=True)
print("TE eui", nrmse(y_eui, oof["eui"], TARGET_COLS[2]), "gas", nrmse(y_gas, oof["eui"] * A, TARGET_COLS[0]))
print("TE elec", nrmse(y_elec, oof["elec"], TARGET_COLS[1]))
np.save("oof/te_eui.npy", oof["eui"]); np.save("oof/te_elec.npy", oof["elec"])
imp = pd.Series(m.feature_importances_, Ta.columns).sort_values(ascending=False)
print(imp.head(25))
