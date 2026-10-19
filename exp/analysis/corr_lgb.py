"""PP0008: nonlinear residual correction. On the holdout chain (VC blend A2xM2xp2 fitted on P-OOF, applied to H), compare
the current linear correction (lstsq on group + kNN features, 5-fold cross-fitted on H) with a small LightGBM fitted on
the residual left by the linear correction (init = linear prediction), features = correction features + log A, log M,
log blend prediction. Both cross-fitted with the same 5 folds of H; LightGBM rounds by early stopping on 20% of each
training part (never on the scored fold).

usage: python exp/analysis/corr_lgb.py
"""
import json
import sys

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
sys.path.insert(0, "exp/analysis")
from corr_est import chain  # noqa: E402
from resid_corr import METRICS  # noqa: E402
from resid_knn import CONFIGS  # noqa: E402
from select_corr import CACHE  # noqa: E402

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=3, seed=0, deterministic=True,
              force_row_wise=True, verbose=-1)


def main():
    tr, te, d, P, H, A, res = chain()
    M = tr["bld_dim_12"].values.astype(float)
    cfgs = {m: [tuple(c) for c in CONFIGS[m]] for m in METRICS}
    cfgs["eui"] = [tuple(c) for c in json.load(open("exp/analysis/corr_final.json"))["eui"]["configs"]]
    folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(H))))
    for metric in METRICS:
        R = res[metric]
        F = np.column_stack([R["G"]] + [np.load(f"{CACHE}/{metric}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in cfgs[metric]])
        X = np.column_stack([F, np.log(A[H]), np.log(M[H]), np.log(np.maximum(R["pH"], 1e-6))])
        r = R["rH"]
        lin, nl = np.zeros(len(H)), np.zeros(len(H))
        rounds = []
        for a, b in folds:
            lam = np.linalg.lstsq(F[a], r[a], rcond=None)[0]
            lin[b] = F[b] @ lam
            base_a = F[a] @ lam
            rng = np.random.default_rng(1)
            es = rng.random(len(a)) < 0.2
            ds = lgb.Dataset(X[a][~es], r[a][~es], init_score=base_a[~es])
            dv = lgb.Dataset(X[a][es], r[a][es], init_score=base_a[es], reference=ds)
            bst = lgb.train(PARAMS, ds, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds.append(bst.best_iteration)
            nl[b] = lin[b] + bst.predict(X[b], num_iteration=bst.best_iteration)
        s = R["sig"]
        base = np.sqrt(np.mean(r ** 2)) / s
        e_lin = np.sqrt(np.mean((r - lin) ** 2)) / s
        e_nl = np.sqrt(np.mean((r - nl) ** 2)) / s
        print(f"{metric}: H blend {base:.5f} linear {e_lin:.5f} linear+lgb {e_nl:.5f} ({e_nl - e_lin:+.6f}) rounds {rounds}",
              flush=True)


if __name__ == "__main__":
    main()
