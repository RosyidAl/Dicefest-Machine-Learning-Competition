"""v8: deeper/slower XGBoost variants of four v4 gas members (max_depth 10, lower learning rate). On the holdout
split (exp_htune.py) swapping them in improved the OOF-weighted gas+eui blend by ~0.0001 total score."""
import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data(n_folds=10)
xp = dict(learning_rate=0.03, max_depth=10, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp")
iA = 1.0 / d.A
run(d, "x8_gpm", "xgb", "gpm", xp, wmul=iA, **TE)
run(d, "x8_gpgea", "xgb", "gas_per:bld_dim_03", xp, wmul=iA, **TE)
run(d, "x8_lgpm", "xgb", "lgas_per:bld_dim_12", xp, wmul=d.A, **TE)
run(d, "x8_gpn", "xgb", "gas_per:bld_dim_01", dict(xp, learning_rate=0.05), wmul=iA, **TE)
