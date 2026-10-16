import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
run(d, "lgb_eui", "lgb", "eui", lp, out_dir="exp")
run(d, "lgb_epm", "lgb", "epm", lp, out_dir="exp")
run(d, "lgb_gas", "lgb", "gas", lp, out_dir="exp")
