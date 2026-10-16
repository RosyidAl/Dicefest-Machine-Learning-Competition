import sys, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold
sys.path.insert(0, "src")
from common import *

tr, te = load()
const = constant_cols(tr)
X = base_features(tr, const)
A = tr["bld_dim_06"].values; M = tr["bld_dim_12"].values
y_gas, y_elec, y_eui = [tr[c].values for c in TARGET_COLS[:3]]
kf = KFold(5, shuffle=True, random_state=SEED)
params = dict(n_estimators=4000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6,
              min_child_weight=5, reg_lambda=1.0, device="cuda", tree_method="hist", random_state=SEED,
              early_stopping_rounds=200, eval_metric="rmse")

def cv(target, weight=None, name=""):
    oof = np.zeros(len(X)); its=[]
    t = time.time()
    for f, (a, b) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[a], target[a], sample_weight=None if weight is None else weight[a],
              eval_set=[(X.iloc[b], target[b])], sample_weight_eval_set=None if weight is None else [weight[b]], verbose=False)
        oof[b] = m.predict(X.iloc[b]); its.append(m.best_iteration)
    print(f"[{name}] iters {its} time {time.time()-t:.0f}s", flush=True)
    return oof

o = cv(y_eui, name="eui")
print("  eui->  eui", nrmse(y_eui, o, TARGET_COLS[2]), " gas", nrmse(y_gas, o*A, TARGET_COLS[0]))
np.save("oof/base_eui.npy", o)
o = cv(y_gas, name="gas")
print("  gas->  eui", nrmse(y_eui, o/A, TARGET_COLS[2]), " gas", nrmse(y_gas, o, TARGET_COLS[0]))
np.save("oof/base_gas.npy", o)
o = cv(y_eui, weight=A**2/np.mean(A**2), name="eui w=A^2")
print("  euiA2->  eui", nrmse(y_eui, o, TARGET_COLS[2]), " gas", nrmse(y_gas, o*A, TARGET_COLS[0]))
o = cv(y_elec, name="elec")
print("  elec", nrmse(y_elec, o, TARGET_COLS[1]))
np.save("oof/base_elec.npy", o)
o = cv(y_elec/M, weight=M**2/np.mean(M**2), name="elec/M w=M^2")
print("  elec/M", nrmse(y_elec, o*M, TARGET_COLS[1]))
o = cv(y_elec/A, weight=A**2/np.mean(A**2), name="elec/A w=A^2")
print("  elec/A", nrmse(y_elec, o*A, TARGET_COLS[1]))
