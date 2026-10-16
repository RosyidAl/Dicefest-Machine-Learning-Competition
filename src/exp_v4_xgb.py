"""v4 zoo, XGBoost part: 10-fold CV with climate-cell encodings."""
import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data(n_folds=10)
xp = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
xf = dict(xp, learning_rate=0.08)  # slow-converging ratio target
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp")
iA, iM = 1.0 / d.A, 1.0 / d.M
run(d, "x4_eui_wA", "xgb", "eui", xp, wmul=d.A, **TE)
run(d, "x4_lepa", "xgb", "lelec_per:bld_dim_06", xp, wmul=d.M.astype(float), **TE)
run(d, "x4_gpn", "xgb", "gas_per:bld_dim_01", xf, wmul=iA, **TE)
run(d, "x4_epm_wM", "xgb", "epm", xp, wmul=iM, **TE)
run(d, "x4_lgpm", "xgb", "lgas_per:bld_dim_12", xp, wmul=d.A, **TE)
run(d, "x4_elec", "xgb", "elec", xp, **TE)
run(d, "x4_leui_wA", "xgb", "leui", xp, wmul=d.A, **TE)
run(d, "x4_epa", "xgb", "elec_per:bld_dim_06", xp, wmul=iM, **TE)
run(d, "x4_gpm", "xgb", "gpm", xp, wmul=iA, **TE)
run(d, "x4_lepm", "xgb", "lepm", xp, wmul=d.M.astype(float), **TE)
run(d, "x4_gas", "xgb", "gas", xp, **TE)
run(d, "x4_gpgea", "xgb", "gas_per:bld_dim_03", xp, wmul=iA, **TE)
