"""FB12: null-importance feature screening (Actual Impurity Reduction vs permuted-target nulls) for LightGBM log EUI
on FS_legacy + agg + TE, fold 0 fit part, 120k-row sample, 10 permutations. Features whose actual gain is below the
75th percentile of their null gains are flagged; configs E0233 then drops them (screening A/B vs E0220)."""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import load  # noqa: E402
from train_cv import FeatureBuilder  # noqa: E402

tr, te = load()
fold = pd.read_parquet("data/folds/folds_primary_k5_s42.parquet")["fold"].values
fb = FeatureBuilder({"set": "legacy", "blocks": ["agg", "te"]}, tr, te)
a, b = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
Xa, _, _ = fb.fold(a, b)
rng = np.random.default_rng(42)
s = np.sort(rng.choice(len(a), 120000, replace=False))
X, y = Xa.iloc[s].reset_index(drop=True), np.log(tr["target_eui_gas_kwh_per_sqm"].values[a][s])
P = dict(objective="regression", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, seed=42, deterministic=True, force_row_wise=True, num_threads=8, verbose=-1)


def gain(yv):
    m = lgb.train(P, lgb.Dataset(X, yv), 300)
    return pd.Series(m.feature_importance("gain"), X.columns)


actual = gain(y)
nulls = pd.concat([gain(rng.permutation(y)) for _ in range(10)], axis=1)
thr = nulls.quantile(0.75, axis=1)
flag = sorted(actual.index[actual <= thr])
out = {"n_features": len(actual), "n_flagged": len(flag), "flagged": flag,
       "top_actual": actual.sort_values(ascending=False).head(15).round(1).to_dict()}
json.dump(out, open("exp/analysis/null_importance.json", "w"), indent=1)
print(len(flag), "of", len(actual), "features flagged:", flag[:40])
