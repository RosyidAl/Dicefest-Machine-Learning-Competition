"""Blend of adoption-model OOF predictions (exp/adopt/<name>_oof.npy, r in percentage points per labelled subzone).

Non-negative least squares on the unseen-weighted squared error (each labelled subzone weighted so that the nrow
buckets follow the row mix of the unseen test subzones), scored honestly with weights fitted on 4/5 of the subzones.

usage: python src/adopt_blend.py name1,name2,...   (e.g. base_q_xent,base_r_reg,cknn_q_xent)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from adopt_v2 import UNSEEN_MIX, cv_scores, subzone_table  # noqa: E402
from common import SEED, load  # noqa: E402


def row_weights(L):
    b = np.minimum(L.nrow.values, 6)
    have = pd.Series(b).value_counts()
    return np.array([UNSEEN_MIX[k] / have[k] for k in b])


def fit_w(R, y, w):
    s = np.sqrt(w)
    return nnls(R * s[:, None], y * s)[0]


def main(names):
    tr, te = load()
    S = subzone_table(tr, te)
    L = S[S.ad.notna()]
    R = np.column_stack([np.load(f"exp/adopt/{n}_oof.npy") for n in names])
    y = (L.rest - L.ad).values  # true r
    w = row_weights(L)
    for j, n in enumerate(names):
        print(f"{n}: unseen-weighted {cv_scores(L, R[:, j])[1]:.4f}")
    wf = fit_w(R, y, w)
    honest = np.zeros(len(L))
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(R):
        honest[b] = R[b] @ fit_w(R[a], y[a], w[a])
    print("weights", dict(zip(names, np.round(wf, 3))), "sum", round(float(wf.sum()), 3))
    print(f"blend in-sample {cv_scores(L, R @ wf)[1]:.4f}, honest {cv_scores(L, honest)[1]:.4f}, "
          f"plain mean {cv_scores(L, R.mean(1))[1]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1].split(","))
