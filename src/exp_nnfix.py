"""Diagnostic: MLP OOF with a fixed epoch (no validation-based epoch choice) vs best-epoch selection."""
import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data(n_folds=10)
kw = dict(epochs=60, lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))
TE = dict(use_te=True, windows=(2, 10), cells=True)
run_nn(d, "n4f_gpm", "gpm", wmul=1.0 / d.A, fixed_ep=37, **TE, **kw)     # n4_gpm best epochs median ~38
run_nn(d, "n4f_lepm", "lepm", wmul=d.M.astype(float), fixed_ep=18, **TE, **kw)  # n4_lepm median ~19
