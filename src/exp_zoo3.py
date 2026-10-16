import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
xp = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0, eval_metric="rmse")
run(d, "xgb_eui_wA_te", "xgb", "eui", xp, out_dir="exp", wmul=d.A, use_te=True)
run(d, "xgb_eui_wA_tew", "xgb", "eui", xp, out_dir="exp", wmul=d.A, use_te=True, windows=(2, 10))
run(d, "xgb_epm_wM_te", "xgb", "epm", xp, out_dir="exp", wmul=1.0 / d.M, use_te=True)
run(d, "xgb_epm_wM_tew", "xgb", "epm", xp, out_dir="exp", wmul=1.0 / d.M, use_te=True, windows=(2, 10))
