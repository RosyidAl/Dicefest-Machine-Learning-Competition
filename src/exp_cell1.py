import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
xp = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0, eval_metric="rmse")
TE = dict(use_te=True, windows=(2, 10), out_dir="exp", cells=True)
run(d, "xgb_eui_wA_tewc", "xgb", "eui", xp, wmul=d.A, **TE)
run(d, "xgb_epm_wM_tewc", "xgb", "epm", xp, wmul=1.0 / d.M, **TE)
