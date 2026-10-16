"""Pick the MLP schedule length honestly: train E epochs with full one-cycle annealing and score the final model."""
import sys; sys.path.insert(0, "src")
import time
import numpy as np
from common import TARGET_COLS, nrmse
from train_lib import Data, is_gas_kind, is_log_kind, target_spec
from nn import fit_fold, preprocess, set_seed, te_block

d = Data(n_folds=10)
Z, Zt = preprocess(d.X, d.Xt)
kw = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256))
FOLDS = (0, 1, 2)
blocks = {k: te_block(d, *d.folds[k], (2, 10), True) for k in FOLDS}


def scan(kind, wmul, Es):
    y, w, mult = target_spec(d, kind)
    w = wmul / wmul.mean() if w is None else w * wmul / np.mean(w * wmul)
    for E in Es:
        t0 = time.time()
        idx, preds = [], []
        for k in FOLDS:
            a, b = d.folds[k]
            Ta, Tb, Tt = blocks[k]
            set_seed(42 + k)
            fixed = None if E == "best60" else E - 1
            pb, _, ep = fit_fold(np.hstack([Z[a], Ta]), y[a], w[a], np.hstack([Z[b], Tb]), y[b], w[b],
                                 np.hstack([Zt[:10], Tt[:10]]), epochs=60 if E == "best60" else E, fixed_ep=fixed, **kw)
            if is_log_kind(kind):
                pb = np.exp(pb)
            idx.append(b); preds.append(pb * (mult if np.isscalar(mult) else mult[b]))
        b = np.concatenate(idx); p = np.concatenate(preds)
        if is_gas_kind(kind):
            msg = f"eui={nrmse(d.y[TARGET_COLS[2]][b], p, TARGET_COLS[2]):.5f} gas={nrmse(d.y[TARGET_COLS[0]][b], p * d.A[b], TARGET_COLS[0]):.5f}"
        else:
            msg = f"elec={nrmse(d.y[TARGET_COLS[1]][b], p, TARGET_COLS[1]):.5f}"
        print(f"[{kind} E={E}] {msg} {time.time() - t0:.0f}s", flush=True)


scan("lepm", d.M.astype(float), [30, 36])
scan("gpm", 1.0 / d.A, [52, 60])
