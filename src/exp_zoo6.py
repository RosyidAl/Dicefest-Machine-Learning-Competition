import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
TE = dict(use_te=True, windows=(2, 10), out_dir="exp")
# LightGBM-like leaf-wise trees on GPU
xl = dict(learning_rate=0.04, grow_policy="lossguide", max_depth=0, max_leaves=255, min_child_weight=100,
          reg_lambda=10.0, subsample=0.8, colsample_bytree=0.5, max_bin=255, eval_metric="rmse")
run(d, "xgbl_eui_wA_tew", "xgb", "eui", xl, wmul=d.A, **TE)
run(d, "xgbl_gpm_tew", "xgb", "gpm", xl, wmul=1.0 / d.A, **TE)
run(d, "xgbl_epm_wM_tew", "xgb", "epm", xl, wmul=1.0 / d.M, **TE)
