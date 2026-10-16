"""H01: Optuna TPE study for LightGBM gas-per-meter (gpm, weight M^2/A; gas NRMSE + derived EUI NRMSE as objective)
on FS_legacy + agg + TE. Search mode: lr 0.1, folds 0-1 of folds_primary_k5_s42 on a 50% row subsample, fold
features computed once. Storage optuna/lgb_gpm.db. The best 3 trials are then confirmed at lr 0.04 on folds 0-2
(full rows) against the reference parameters. Output: exp/analysis/optuna_lgb.json, reports/phases/hpo/*.png"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS, load  # noqa: E402
from train_cv import FeatureBuilder  # noqa: E402

os.makedirs("optuna", exist_ok=True)
os.makedirs("reports/phases/hpo", exist_ok=True)
tr, te = load()
fold = pd.read_parquet("data/folds/folds_primary_k5_s42.parquet")["fold"].values
fb = FeatureBuilder({"set": "legacy", "blocks": ["agg", "te"]}, tr, te)
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
yg, ye = tr[TARGET_COLS[0]].values, tr[TARGET_COLS[2]].values
rng = np.random.default_rng(42)
sub = rng.random(len(tr)) < 0.5
cache = {}


def fold_data(k, subsample):
    key = (k, subsample)
    if key not in cache:
        m = sub if subsample else np.ones(len(tr), bool)
        a, b = np.flatnonzero((fold != k) & m), np.flatnonzero((fold == k) & m)
        Xa, Xb, _ = fb.fold(a, b)
        cache[key] = (a, b, Xa, Xb)
    return cache[key]


def score(params, folds, subsample, lr, trial=None):
    tot = []
    for i, k in enumerate(folds):
        a, b, Xa, Xb = fold_data(k, subsample)
        w = M ** 2 / A
        p = dict(objective="regression", learning_rate=lr, seed=42, deterministic=True, force_row_wise=True,
                 num_threads=8, verbose=-1, **params)
        da = lgb.Dataset(Xa, yg[a] / M[a], weight=w[a] / w[a].mean(), free_raw_data=False)
        db = lgb.Dataset(Xb, yg[b] / M[b], weight=w[b] / w[b].mean(), reference=da)
        m = lgb.train(p, da, 20000, valid_sets=[db], callbacks=[lgb.early_stopping(200, verbose=False)])
        pg = m.predict(Xb, num_iteration=m.best_iteration) * M[b]
        s = (np.sqrt(np.mean((pg - yg[b]) ** 2)) / SIGMAS[TARGET_COLS[0]] +
             np.sqrt(np.mean((pg / A[b] - ye[b]) ** 2)) / SIGMAS[TARGET_COLS[2]])
        tot.append(s)
        if trial is not None:
            trial.report(float(np.mean(tot)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(tot))


REF = dict(num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=10.0, max_bin=255)


def objective(trial):
    p = dict(num_leaves=trial.suggest_int("num_leaves", 31, 2047, log=True),
             min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 1000, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.2, 1.0),
             bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0), bagging_freq=1,
             lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
             lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 100, log=True),
             min_gain_to_split=trial.suggest_float("min_gain_to_split", 0, 1),
             path_smooth=trial.suggest_float("path_smooth", 0, 20),
             max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511]))
    return score(p, [0, 1], True, 0.1, trial)


study = optuna.create_study(study_name="lgb_gpm", storage="sqlite:///optuna/lgb_gpm.db", load_if_exists=True,
                            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=0))
study.enqueue_trial({k: v for k, v in REF.items() if k != "bagging_freq"} | dict(lambda_l1=1e-8, min_gain_to_split=0.0, path_smooth=0.0))
n_left = 40 - len([t for t in study.trials if t.state.is_finished()])
if n_left > 0:
    study.optimize(objective, n_trials=n_left)
done = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
res = {"n_trials": len(study.trials), "best_search": [{"value": t.value, "params": t.params} for t in done[:3]]}
res["confirm_ref"] = score(REF, [0, 1, 2], False, 0.04)
res["confirm"] = []
for t in done[:3]:
    p = dict(t.params, bagging_freq=1)
    res["confirm"].append({"params": p, "value": score(p, [0, 1, 2], False, 0.04)})
    print(res["confirm"][-1], "ref", res["confirm_ref"], flush=True)
json.dump(res, open("exp/analysis/optuna_lgb.json", "w"), indent=1)
try:
    import optuna.visualization.matplotlib as ov
    import matplotlib.pyplot as plt
    ov.plot_optimization_history(study); plt.savefig("reports/phases/hpo/lgb_gpm_history.png", dpi=150); plt.close()
    ov.plot_param_importances(study); plt.savefig("reports/phases/hpo/lgb_gpm_importance.png", dpi=150); plt.close()
except Exception as e:  # noqa: BLE001
    print("plot failed", e)
print(json.dumps(res, indent=1))
