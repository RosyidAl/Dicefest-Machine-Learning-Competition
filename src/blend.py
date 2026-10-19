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


def size_basis(a_fit, a, knots):
    """Hat-function basis on the mid-rank of `a` within `a_fit` (the fit rows' empirical CDF, no target involved):
    one column per knot in [0, 1], rows sum to 1. Used for member weights that vary smoothly with building size."""
    s = np.sort(a_fit)
    t = (np.searchsorted(s, a, side="left") + np.searchsorted(s, a, side="right")) / (2.0 * len(s))
    k = np.asarray(knots, dtype=float)
    B = np.zeros((len(t), len(k)))
    for j in range(len(k)):
        b = np.zeros(len(t))
        if j == 0:
            b[t <= k[0]] = 1.0
        else:
            m = (t >= k[j - 1]) & (t <= k[j])
            b[m] = (t[m] - k[j - 1]) / (k[j] - k[j - 1])
        if j == len(k) - 1:
            b[t >= k[j]] = 1.0
        else:
            m = (t > k[j]) & (t <= k[j + 1])
            b[m] = (k[j + 1] - t[m]) / (k[j + 1] - k[j])
        B[:, j] = b
    return B


def nnls_blend_vc(P, y, B, row_scale=None):
    """Size-varying NNLS blend: weights W (members x knots) >= 0; prediction sum_j B[:, j] * (P @ W[:, j])."""
    X = np.concatenate([P * B[:, [j]] for j in range(B.shape[1])], axis=1)
    return nnls_blend(X, y, row_scale).reshape(B.shape[1], P.shape[1]).T


def predict_vc(P, W, B):
    return ((P @ W) * B).sum(axis=1)
