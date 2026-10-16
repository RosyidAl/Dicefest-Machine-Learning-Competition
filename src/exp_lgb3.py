import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
TE = dict(use_te=True, windows=(2, 10), out_dir="exp")
iA, iM = 1.0 / d.A, 1.0 / d.M
run(d, "lgb_gpm_tew", "lgb", "gpm", lp, wmul=iA, **TE)
run(d, "lgb_gpgea_tew", "lgb", "gas_per:bld_dim_03", lp, wmul=iA, **TE)
run(d, "lgb_gpn_tew", "lgb", "gas_per:bld_dim_01", lp, wmul=iA, **TE)
run(d, "lgb_elec_tew", "lgb", "elec", lp, **TE)
run(d, "lgb_epa_tew", "lgb", "elec_per:bld_dim_06", lp, wmul=iM, **TE)
run(d, "lgb_lgpm_tew", "lgb", "lgas_per:bld_dim_12", lp, wmul=d.A, **TE)
