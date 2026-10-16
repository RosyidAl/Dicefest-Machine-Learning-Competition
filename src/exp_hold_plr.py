"""MLPs with PLR numerical embeddings (nn.PLR) on the holdout split of exp_hold.py. Outputs go to
exp/hold/tune/{oof,pred}; compare with the plain-MLP hold members n6_*.

usage: python src/exp_hold_plr.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from exp_hold import W, nk, s  # noqa: E402
from nn import run_nn  # noqa: E402

EMB = dict(k=24, sigma=0.05, d=24)
# name, kind, epochs (as the n6_* base), weight key, embedding settings
JOBS = [
    ("p1_lepm", "lepm", 24, "M", EMB),
    ("p1_gpm", "gpm", 44, "iA", EMB),
    ("p1_epm_wM", "epm", 23, "iM", EMB),
]
for name, kind, ep, wk, emb in JOBS:
    if not os.path.exists(f"exp/hold/tune/pred/{name}.npy"):
        run_nn(s, name, kind, out_dir="exp/hold/tune", wmul=W[wk], use_te=True, windows=(2, 10), cells=True,
               epochs=ep, fixed_ep=ep - 1, emb=emb, **nk)
