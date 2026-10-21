"""Out-of-fold group target statistics (hierarchical spatial encodings)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def group_stats(fit_keys, fit_vals, apply_keys):
    """Sum and count of `fit_vals` columns per key, looked up for `apply_keys`.

    fit_vals: 2-D array (n_fit, k). Returns (sums (n_apply, k), counts (n_apply,)).
    """
    df = pd.DataFrame(fit_vals)
    df["_k"] = fit_keys
    g = df.groupby("_k", sort=False)
    sums = g.sum()
    cnt = g.size()
    idx = pd.Index(sums.index).get_indexer(apply_keys)
    out_s = np.zeros((len(apply_keys), fit_vals.shape[1]))
    out_c = np.zeros(len(apply_keys))
    ok = idx >= 0
    out_s[ok] = sums.values[idx[ok]]
    out_c[ok] = cnt.values[idx[ok]]
    return out_s, out_c


def oof_group_stats(keys, vals, n_splits=5, seed=42):
    """K-fold out-of-fold sums/counts for every row of one set."""
    s = np.zeros((len(keys), vals.shape[1]))
    c = np.zeros(len(keys))
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(keys):
        s[b], c[b] = group_stats(keys[a], vals[a], keys[b])
    return s, c


def window_stats(fit_pos, fit_vals, apply_pos, w, size):
    """Sums/counts over fit rows whose position lies in [p-w, p+w], excluding position p itself.

    Positions are the numeric subzone ids, which are ordered roughly geographically.
    """
    k = fit_vals.shape[1]
    S = np.zeros((size + 2 * w + 2, k))
    C = np.zeros(size + 2 * w + 2)
    np.add.at(S, fit_pos + w + 1, fit_vals)
    np.add.at(C, fit_pos + w + 1, 1.0)
    cS = np.cumsum(S, axis=0)
    cC = np.cumsum(C)
    hi = apply_pos + 2 * w + 1
    lo = apply_pos
    s = cS[hi] - cS[lo] - S[apply_pos + w + 1]
    c = cC[hi] - cC[lo] - C[apply_pos + w + 1]
    return s, c


def oof_window_stats(pos, vals, w, size, n_splits=5, seed=42):
    s = np.zeros((len(pos), vals.shape[1]))
    c = np.zeros(len(pos))
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(pos):
        s[b], c[b] = window_stats(pos[a], vals[a], pos[b], w, size)
    return s, c


def stats_to_features(s, c, names, prefix, prior=None, k=0.0):
    """Turn sums/counts into (optionally shrunk) means plus a count column."""
    out = {f"{prefix}_cnt": c}
    for j, n in enumerate(names):
        if prior is None:
            m = np.where(c > 0, s[:, j] / np.maximum(c, 1), np.nan)
        else:
            m = (s[:, j] + k * prior[j]) / (c + k)
        out[f"{prefix}_{n}"] = m
    return pd.DataFrame(out)
