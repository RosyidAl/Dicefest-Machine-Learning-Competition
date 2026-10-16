"""v13 test: XGBoost members without the subzone target encodings (sz_* columns), for test rows of subzones unseen
in train (8% of test). The regular models learn the no-subzone-information regime only from the ~10% of training
rows whose subzone has no other rows in the encoding set; these models learn it from every row.

usage: python src/exp_v13_xs.py name1,name2,...
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
NO_SZ = tuple(["sz_cnt"] + [f"sz_{n}" for n in d.VN + d.WN])
iA, iM = 1.0 / d.A, 1.0 / d.M
# name: (target formulation, params, sample weight)
JOBS = {
    "xs_gpm": ("gpm", x8, iA),
    "xs_epm_wM": ("epm", x4, iM),
}
for name in sys.argv[1].split(","):
    kind, params, wmul = JOBS[name]
    if not os.path.exists(f"exp/pred/{name}.npy"):
        run(d, name, "xgb", kind, params, wmul=wmul, drop=NO_SZ, **TE)
