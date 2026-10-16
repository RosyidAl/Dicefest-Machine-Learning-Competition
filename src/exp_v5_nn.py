"""v5: second seed set for the heaviest-weighted MLPs (10-fold, climate cells)."""
import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data(n_folds=10)
kw = dict(epochs=60, lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(45, 46, 47))
TE = dict(use_te=True, windows=(2, 10), cells=True)
iA, iM = 1.0 / d.A, 1.0 / d.M
run_nn(d, "n5_gpm", "gpm", wmul=iA, **TE, **kw)
run_nn(d, "n5_epm_wM", "epm", wmul=iM, **TE, **kw)
run_nn(d, "n5_lepm", "lepm", wmul=d.M.astype(float), **TE, **kw)
run_nn(d, "n5_gpgea", "gas_per:bld_dim_03", wmul=iA, **TE, **kw)
