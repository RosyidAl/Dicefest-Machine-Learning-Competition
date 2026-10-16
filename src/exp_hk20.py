"""20 folds instead of 10 inside P (each model sees 95% instead of 90% of P), judged on the holdout H.
Compare exp/hold/tune/pred/<name>_k20 with the 10-fold exp/hold/pred/<name>.

usage: python src/exp_hk20.py name1,name2,...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from exp_hold import CPU, GPU, H, P, TE, W, d, subset  # noqa: E402
from train_lib import run  # noqa: E402

s20 = subset(d, P, H, k=20)
spec = {r[0]: r for r in GPU + CPU}
for name in sys.argv[1].split(","):
    _, backend, kind, params, wk = spec[name]
    if not os.path.exists(f"exp/hold/tune/pred/{name}_k20.npy"):
        run(s20, f"{name}_k20", backend, kind, params, wmul=W[wk], verbose=False, **dict(TE, out_dir="exp/hold/tune"))
