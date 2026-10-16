"""v4 zoo, MLP part: 10-fold CV, climate-cell encodings, 3 seeds per fold."""
import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data(n_folds=10)
kw = dict(epochs=60, lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))
TE = dict(use_te=True, windows=(2, 10), cells=True)
iA, iM = 1.0 / d.A, 1.0 / d.M
run_nn(d, "n4_gpm", "gpm", wmul=iA, **TE, **kw)
run_nn(d, "n4_lepm", "lepm", wmul=d.M.astype(float), **TE, **kw)
run_nn(d, "n4_leui_wA", "leui", wmul=d.A, **TE, **kw)
run_nn(d, "n4_epm_wM", "epm", wmul=iM, **TE, **kw)
run_nn(d, "n4_eui_wA", "eui", wmul=d.A, **TE, **kw)
run_nn(d, "n4_gpgea", "gas_per:bld_dim_03", wmul=iA, **TE, **kw)
run_nn(d, "n4_lgpm", "lgas_per:bld_dim_12", wmul=d.A, **TE, **kw)
run_nn(d, "n4_epa", "elec_per:bld_dim_06", wmul=iM, **TE, **kw)
