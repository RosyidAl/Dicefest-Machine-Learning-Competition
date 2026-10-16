"""Extra MLP runs on the holdout split of exp_hold.py: n6b_* = the n6_* configs with seeds 45-47 (same folds),
n6r_* = seeds 42-44 on a different 10-fold split. Compares seed averaging with fold-split averaging."""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import KFold  # noqa: E402
from exp_hold import GPU, W, nk, s  # noqa: E402
from nn import run_nn  # noqa: E402

NN = [r for r in GPU if r[1] == "nn"]
s2 = copy.copy(s)
s2.folds = list(KFold(10, shuffle=True, random_state=43).split(s.X))
jobs = []
for name, _, kind, ep, wk in sorted(NN, key=lambda r: r[0] not in ("n6_gpm", "n6_lepm")):
    jobs.append((s, name.replace("n6_", "n6b_"), kind, ep, wk, (45, 46, 47)))
    if name in ("n6_gpm", "n6_lepm"):
        jobs.append((s2, name.replace("n6_", "n6r_"), kind, ep, wk, (42, 43, 44)))
for data, name, kind, ep, wk, seeds in jobs:
    if not os.path.exists(f"exp/hold/pred/{name}.npy"):
        run_nn(data, name, kind, out_dir="exp/hold", wmul=W[wk], use_te=True, windows=(2, 10), cells=True,
               epochs=ep, fixed_ep=ep - 1, **dict(nk, seeds=seeds))
