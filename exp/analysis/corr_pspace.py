"""PP0009: kNN residual-correction features in prediction space. Neighbours within +-W sub_n whose blend predictions
are closest (space 'p': standardised log blend prediction; space 'pf': building-stock space 'full' plus the log
prediction), same estimator per metric as the current kNN features. Holdout chain as corr_est.py; honest split-half
check of greedy additions on top of the current configs.

usage: python exp/analysis/corr_pspace.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "exp/analysis")
from corr_est import chain, estimator, greedy, knn_multi  # noqa: E402
from resid_corr import METRICS  # noqa: E402
from resid_knn import spaces  # noqa: E402
from select_corr import CACHE, cv_rmse  # noqa: E402

GRIDP = [(sp, W, k) for sp in ("p", "pf") for W, k in ((10, 10), (30, 10), (100, 20), (300, 50), (1000, 100))]


def main():
    tr, te, d, P, H, A, res = chain()
    Zs = spaces(tr, te)
    n = len(tr)
    cfgs = {m: [tuple(c) for c in v] for m, v in json.load(open("exp/resid_knn_vc_AMp_e.json"))["configs"].items()}
    halves = np.random.default_rng(11).permutation(len(H)) % 2
    for metric in METRICS:
        R = res[metric]
        lp = np.log(np.maximum(np.r_[R["pP"], R["pH"]], 1e-6))
        lp = (lp - lp[: len(P)].mean()) / lp[: len(P)].std()
        full = Zs["full"][:n]
        Zp = {"p": lp[:, None], "pf": np.column_stack([np.r_[full[P], full[H]], lp])}
        num, den, mult = estimator("base", metric, R["rP"], R["pP"], R["pH"], A[P], A[H])
        cols = {}
        for sp, W, k in GRIDP:
            f = f"{CACHE}/psp_{metric}_{sp}_{W}_{k}.npy"
            if not os.path.exists(f):
                Z = Zp[sp]
                v = knn_multi(Z[: len(P)], d.subn[P], Z[len(P):], d.subn[H], num[:, None], den[:, None], W, k)[:, 0]
                np.save(f, v * mult)
            cols[(sp, W, k)] = np.load(f)
        F0 = np.column_stack([R["G"]] + [np.load(f"{CACHE}/{metric}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in cfgs[metric]])
        out = []
        for h in (0, 1):
            a, b = halves == h, halves != h
            sel = greedy(F0[a], {c: v[a] for c, v in cols.items()}, R["rH"][a], R["sig"])
            new = cv_rmse(np.column_stack([F0[b]] + [cols[c][b] for c in sel]), R["rH"][b], R["sig"])
            old = cv_rmse(F0[b], R["rH"][b], R["sig"])
            out.append(new - old)
            print(f"{metric} half {h}: +{sel} -> {new:.5f} vs {old:.5f} ({new - old:+.6f})", flush=True)
        allsel = greedy(F0, cols, R["rH"], R["sig"])
        print(f"{metric} mean honest change {np.mean(out):+.6f}; on all H: +{allsel}", flush=True)


if __name__ == "__main__":
    main()
