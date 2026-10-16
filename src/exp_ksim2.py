"""Pseudo-test simulation, part 2: same 70/30 split as exp_ksim.py, 10 folds, but encodings for validation rows
use all other labelled rows (leave-one-out) and holdout rows use all labelled rows (te_full=True)."""
import sys; sys.path.insert(0, "src")
import numpy as np
from common import TARGET_COLS, nrmse
from train_lib import run
from exp_ksim import d, P, H, subset, xp  # noqa: E402  (module guard below keeps its runs from executing)

TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp/sim", verbose=False, te_full=True)
s = subset(d, P, H, 10)
oof, pt = run(s, "sim_eui_wA_k10_full", "xgb", "eui", xp, wmul=s.A, **TE)
print(f"   k=10 full-TE eui_wA: OOF eui {nrmse(s.y[TARGET_COLS[2]], oof, TARGET_COLS[2]):.5f} "
      f"HOLDOUT eui {nrmse(d.y[TARGET_COLS[2]][H], pt, TARGET_COLS[2]):.5f} "
      f"OOF gas {nrmse(s.y[TARGET_COLS[0]], oof * s.A, TARGET_COLS[0]):.5f} "
      f"HOLDOUT gas {nrmse(d.y[TARGET_COLS[0]][H], pt * d.A[H], TARGET_COLS[0]):.5f}", flush=True)
