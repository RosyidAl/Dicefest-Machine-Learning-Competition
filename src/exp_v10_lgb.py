"""v10, CPU part: LightGBM v4 members with the subzone / zone aggregates over train+test rows
(Data.agg_features). 5-fold A/B for LightGBM gpm: EUI 0.66575 -> 0.66465, gas 0.25354 -> 0.25322 (richer
context features - subzone means of all postcode-level columns, cell aggregates - added nothing more).

usage: python src/exp_v10_lgb.py name1,name2,...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from train_lib import Data, run  # noqa: E402

d = Data(n_folds=10)
ex, ext = d.agg_features()
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp", extra=ex, extra_t=ext)
iA, iM = 1.0 / d.A, 1.0 / d.M
# name: (target formulation, sample weight)
JOBS = {
    "la_gpm": ("gpm", iA),
    "la_eui_wA": ("eui", d.A),
    "la_epm_wM": ("epm", iM),
    "la_lgpm": ("lgas_per:bld_dim_12", d.A),
    "la_leui_wA": ("leui", d.A),
    "la_lepm": ("lepm", d.M.astype(float)),
    "la_epa": ("elec_per:bld_dim_06", iM),
    "la_elec": ("elec", None),
    "la_gas": ("gas", None),
    "la_lepa": ("lelec_per:bld_dim_06", d.M.astype(float)),
    "la_gpgea": ("gas_per:bld_dim_03", iA),
}
for name in sys.argv[1].split(","):
    kind, wmul = JOBS[name]
    if not os.path.exists(f"exp/pred/{name}.npy"):
        run(d, name, "lgb", kind, lp, wmul=wmul, **TE)
