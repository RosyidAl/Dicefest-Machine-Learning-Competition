"""v10: XGBoost members with subzone / zone aggregates over train+test rows (Data.agg_features) as extra
features. 5-fold A/B for XGB gpm: gas 0.25354 -> 0.25317, EUI 0.66497 -> 0.66476.

usage: python src/exp_v10_agg.py name1,name2,...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from train_lib import Data, run  # noqa: E402

d = Data(n_folds=10)
ex, ext = d.agg_features()
x8 = dict(learning_rate=0.03, max_depth=10, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
x4 = dict(x8, learning_rate=0.05, max_depth=8)
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp", extra=ex, extra_t=ext)
iA, iM = 1.0 / d.A, 1.0 / d.M
# name: (target formulation, params, sample weight)
JOBS = {
    "xa_gpm": ("gpm", x8, iA),
    "xa_gpgea": ("gas_per:bld_dim_03", x8, iA),
    "xa_lgpm": ("lgas_per:bld_dim_12", x8, d.A),
    "xa_lepa": ("lelec_per:bld_dim_06", x4, d.M.astype(float)),
    "xa_epm_wM": ("epm", x4, iM),
}
for name in sys.argv[1].split(","):
    kind, params, wmul = JOBS[name]
    if not os.path.exists(f"exp/pred/{name}.npy"):
        run(d, name, "xgb", kind, params, wmul=wmul, **TE)
