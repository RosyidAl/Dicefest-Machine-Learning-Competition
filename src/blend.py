"""Non-negative least-squares blending of saved OOF predictions, one weight vector per scored column."""
import numpy as np
from scipy.optimize import nnls


def nnls_blend(P, y, row_scale=None):
    """Weights w >= 0 minimising ||s*(y - P w)|| where s is an optional per-row scale (e.g. floor area)."""
    if row_scale is None:
        row_scale = np.ones(len(y))
    w, _ = nnls(P * row_scale[:, None], y * row_scale)
    return w


def cv_blend_score(P, y, row_scale=None, n_splits=5, seed=0):
    """Honest blend score: weights fitted on 4/5 of rows, scored on the remaining 1/5."""
    from sklearn.model_selection import KFold
    s = np.ones(len(y)) if row_scale is None else row_scale
    pred = np.zeros(len(y))
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(P):
        w = nnls_blend(P[a], y[a], s[a])
        pred[b] = P[b] @ w
    return pred
