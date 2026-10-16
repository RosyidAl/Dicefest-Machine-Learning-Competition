import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
TE = dict(use_te=True, windows=(2, 10), out_dir="exp")
run(d, "lgb_eui_wA_tew", "lgb", "eui", lp, wmul=d.A, **TE)
run(d, "lgb_epm_wM_tew", "lgb", "epm", lp, wmul=1.0 / d.M, **TE)
run(d, "lgb_leui_wA_tew", "lgb", "leui", lp, wmul=d.A, **TE)
run(d, "lgb_lepm_tew", "lgb", "lepm", lp, wmul=d.M.astype(float), **TE)
