"""Plain MLPs (the n6_* configs) with a different mini-batch order per seed and fold (nn.run_nn vary_order),
on the holdout split of exp_hold.py. Outputs in exp/hold/tune; compare with n6_*.

usage: python src/exp_hold_order.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from exp_hold import W, nk, s  # noqa: E402
from nn import run_nn  # noqa: E402

JOBS = [("no_lepm", "lepm", 24, "M"), ("no_gpm", "gpm", 44, "iA")]
for name, kind, ep, wk in JOBS:
    if not os.path.exists(f"exp/hold/tune/pred/{name}.npy"):
        run_nn(s, name, kind, out_dir="exp/hold/tune", wmul=W[wk], use_te=True, windows=(2, 10), cells=True,
               epochs=ep, fixed_ep=ep - 1, vary_order=True, **nk)
