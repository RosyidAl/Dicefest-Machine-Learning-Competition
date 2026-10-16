"""v5: more CatBoost formulations (10-fold, climate cells)."""
import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data(n_folds=10)
cp = dict(learning_rate=0.08, depth=8, l2_leaf_reg=3, border_count=254)
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp")
iA, iM = 1.0 / d.A, 1.0 / d.M
run(d, "c4_lgpm", "cat", "lgas_per:bld_dim_12", cp, wmul=d.A, **TE)
run(d, "c4_lepa", "cat", "lelec_per:bld_dim_06", cp, wmul=d.M.astype(float), **TE)
run(d, "c4_gpgea", "cat", "gas_per:bld_dim_03", cp, wmul=iA, **TE)
run(d, "c4_epa", "cat", "elec_per:bld_dim_06", cp, wmul=iM, **TE)
run(d, "c4_leui_wA", "cat", "leui", cp, wmul=d.A, **TE)
