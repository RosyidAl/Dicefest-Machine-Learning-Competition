import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
base = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_lambda=1.0, eval_metric="rmse")
grid = {
    "xgb_eui_d6": dict(base, max_depth=6),
    "xgb_eui_d10": dict(base, max_depth=10, min_child_weight=20),
    "xgb_eui_d8_mcw50": dict(base, min_child_weight=50),
    "xgb_eui_d8_cs03": dict(base, colsample_bytree=0.3),
    "xgb_eui_lg": dict(base, grow_policy="lossguide", max_depth=0, max_leaves=255, min_child_weight=20),
}
for name, p in grid.items():
    run(d, name, "xgb", "eui", p, out_dir="exp")
