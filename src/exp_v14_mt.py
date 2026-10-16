"""v14: multi-task PLR MLP - one network fits gas per meter (gpm, weight M^2/A) and elec per meter (epm, weight M)
with a shared trunk. Fold-0 test (1 seed, 12 epochs): with hidden (2048, 1024, 512) the multi-task net beat the
single-task nets of the same size on all metrics (EUI 0.66743 vs 0.66889, gas 0.25142 vs 0.25190, elec 0.32029 vs
0.32133); with (1024, 512, 256) the result was mixed.

usage: python src/exp_v14_mt.py name1,name2,...   -> exp/oof|pred/<name>_gpm.npy and <name>_epm.npy
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from train_lib import Data  # noqa: E402
from nn import run_nn_multi  # noqa: E402

d = Data(n_folds=10)
ex, ext = d.agg_features()
TE = dict(use_te=True, windows=(2, 10), cells=True, extra=ex, extra_t=ext)
EMB = dict(k=24, sigma=0.05, d=24)
MT = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(2048, 1024, 512), seeds=(42, 43, 44), emb=EMB)
iA, iM = 1.0 / d.A, 1.0 / d.M
# name: (tasks [(suffix, kind, weight)], epochs, model settings)
JOBS = {
    "m1": ([("gpm", "gpm", iA), ("epm", "epm", iM)], 12, MT),
}
for name in sys.argv[1].split(","):
    tasks, E, cfg = JOBS[name]
    if not all(os.path.exists(f"exp/pred/{name}_{t[0]}.npy") for t in tasks):
        run_nn_multi(d, name, tasks, epochs=E, fixed_ep=E - 1, **TE, **cfg)
