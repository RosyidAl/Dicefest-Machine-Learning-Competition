import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
xp = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0, eval_metric="rmse")
run(d, "xgb_eui_wA", "xgb", "eui", xp, out_dir="exp", wmul=d.A)
run(d, "xgb_leui_wA", "xgb", "leui", xp, out_dir="exp", wmul=d.A)
run(d, "xgb_gas_wA", "xgb", "gas", xp, out_dir="exp", wmul=1.0 / d.A)
run(d, "xgb_epm_wM", "xgb", "epm", xp, out_dir="exp", wmul=1.0 / d.M)
run(d, "xgb_lepm", "xgb", "lepm", xp, out_dir="exp", wmul=d.M.astype(float))
