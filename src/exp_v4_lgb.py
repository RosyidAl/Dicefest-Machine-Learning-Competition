"""v4 zoo, CPU part: LightGBM with 10-fold CV and climate-cell encodings."""
import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data(n_folds=10)
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp")
iA, iM = 1.0 / d.A, 1.0 / d.M
run(d, "l4_gpm", "lgb", "gpm", lp, wmul=iA, **TE)
run(d, "l4_epm_wM", "lgb", "epm", lp, wmul=iM, **TE)
run(d, "l4_eui_wA", "lgb", "eui", lp, wmul=d.A, **TE)
run(d, "l4_lepm", "lgb", "lepm", lp, wmul=d.M.astype(float), **TE)
run(d, "l4_leui_wA", "lgb", "leui", lp, wmul=d.A, **TE)
run(d, "l4_lgpm", "lgb", "lgas_per:bld_dim_12", lp, wmul=d.A, **TE)
