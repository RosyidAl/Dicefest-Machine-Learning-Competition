"""Noise-matched weights: residual log-variance is a + b/M (gas a/b~0.024, elec a/b~0.019), so the inverse
variance grows like M/(1+cM) rather than M. Same LightGBM setup as v4, weights damped for large postcodes."""
import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data(n_folds=10)
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp")
M = d.M.astype(float)
dg, de = 1.0 / (1 + 0.024 * M), 1.0 / (1 + 0.019 * M)
run(d, "l7_lepm_wv", "lgb", "lepm", lp, wmul=M * de, **TE)          # compare l4_lepm 0.32257
run(d, "l7_eui_wv", "lgb", "eui", lp, wmul=d.A * dg, **TE)          # compare l4_eui_wA 0.66262/0.25132
run(d, "l7_gpm_wv", "lgb", "gpm", lp, wmul=dg / d.A, **TE)          # compare l4_gpm 0.66209/0.25165
run(d, "l7_epm_wv", "lgb", "epm", lp, wmul=de / M, **TE)            # compare l4_epm_wM 0.32161
