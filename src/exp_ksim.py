"""Pseudo-test simulation: hold out 30% of train as a labelled 'test', then compare how 5-fold and 10-fold CV
OOF scores relate to the score of the fold-averaged predictions on the held-out rows."""
import sys; sys.path.insert(0, "src")
import copy
import numpy as np
from sklearn.model_selection import KFold
from common import SEED, TARGET_COLS, nrmse
from train_lib import Data, run

d = Data(n_folds=5)
rng = np.random.default_rng(7)
perm = rng.permutation(len(d.X))
P, H = np.sort(perm[: int(0.7 * len(perm))]), np.sort(perm[int(0.7 * len(perm)):])


def subset(d, P, H, k):
    s = copy.copy(d)
    s.X, s.Xt = d.X.iloc[P].reset_index(drop=True), d.X.iloc[H].reset_index(drop=True)
    s.A, s.At, s.M, s.Mt = d.A[P], d.A[H], d.M[P], d.M[H]
    s.y = {c: d.y[c][P] for c in TARGET_COLS}
    s.sub, s.subt, s.zone, s.zonet = d.sub[P], d.sub[H], d.zone[P], d.zone[H]
    s.subn, s.subnt = d.subn[P], d.subn[H]
    s.V, s.W = d.V[P], d.W[P]
    s.cells, s.cellst = [c[P] for c in d.cells], [c[H] for c in d.cells]
    s.folds = list(KFold(k, shuffle=True, random_state=SEED).split(s.X))
    return s


xp = dict(learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse", early_stopping_rounds=100)
if __name__ == "__main__":
    TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp/sim", verbose=False)
    for kind, wm, tag in [("eui", lambda s: s.A, "eui_wA"), ("epm", lambda s: 1.0 / s.M, "epm_wM")]:
        for k in (5, 10):
            s = subset(d, P, H, k)
            oof, pt = run(s, f"sim_{tag}_k{k}", "xgb", kind, xp, wmul=wm(s), **TE)
            if kind == "eui":
                print(f"   k={k} {tag}: OOF eui {nrmse(s.y[TARGET_COLS[2]], oof, TARGET_COLS[2]):.5f} "
                      f"HOLDOUT eui {nrmse(d.y[TARGET_COLS[2]][H], pt, TARGET_COLS[2]):.5f} "
                      f"OOF gas {nrmse(s.y[TARGET_COLS[0]], oof * s.A, TARGET_COLS[0]):.5f} "
                      f"HOLDOUT gas {nrmse(d.y[TARGET_COLS[0]][H], pt * d.A[H], TARGET_COLS[0]):.5f}", flush=True)
            else:
                print(f"   k={k} {tag}: OOF elec {nrmse(s.y[TARGET_COLS[1]], oof, TARGET_COLS[1]):.5f} "
                      f"HOLDOUT elec {nrmse(d.y[TARGET_COLS[1]][H], pt, TARGET_COLS[1]):.5f}", flush=True)
