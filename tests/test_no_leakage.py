"""LK1: target-based feature blocks must not see validation labels. Shuffle the labels of the validation rows of one
fold, recompute the block, and require the validation (and test) features to be bit-identical."""
import numpy as np
import pandas as pd

from common import TARGET_COLS


def _subset(n=20000, seed=0):
    tr = pd.read_parquet("data/cache/train.parquet")
    te = pd.read_parquet("data/cache/test.parquet")
    rng = np.random.default_rng(seed)
    return tr.iloc[np.sort(rng.choice(len(tr), n, replace=False))].reset_index(drop=True), \
        te.iloc[np.sort(rng.choice(len(te), n // 2, replace=False))].reset_index(drop=True)


def test_knn_target_block():
    from features.blocks import KnnTarget
    tr, te = _subset()
    rng = np.random.default_rng(1)
    b = np.sort(rng.choice(len(tr), len(tr) // 5, replace=False))
    a = np.setdiff1d(np.arange(len(tr)), b)
    _, Fb1, Ft1 = KnnTarget(tr, te, configs=((30, 10, 0.0), (100, 10, 4.0))).fold(a, b)
    tr2 = tr.copy()
    for c in TARGET_COLS:
        tr2.loc[b, c] = rng.permutation(tr2.loc[b, c].values)
    _, Fb2, Ft2 = KnnTarget(tr2, te, configs=((30, 10, 0.0), (100, 10, 4.0))).fold(a, b)
    assert Fb1.equals(Fb2) and Ft1.equals(Ft2)


def test_group_target_encoding():
    """train_lib.Data.te_features: validation/test encodings come from the fit part only"""
    import train_lib
    tr, te = _subset(30000)
    orig = train_lib.load
    try:
        train_lib.load = lambda cache="cache": (tr.copy(), te.copy())
        d1 = train_lib.Data()
        rng = np.random.default_rng(2)
        b = np.sort(rng.choice(len(tr), len(tr) // 5, replace=False))
        a = np.setdiff1d(np.arange(len(tr)), b)
        f1 = d1.te_features(a, d1.te_apply(b), (2, 10), True)
        t1 = d1.te_features(a, d1.te_apply(), (2, 10), True)
        tr2 = tr.copy()
        for c in TARGET_COLS:
            tr2.loc[b, c] = rng.permutation(tr2.loc[b, c].values)
        train_lib.load = lambda cache="cache": (tr2.copy(), te.copy())
        d2 = train_lib.Data()
        f2 = d2.te_features(a, d2.te_apply(b), (2, 10), True)
        t2 = d2.te_features(a, d2.te_apply(), (2, 10), True)
    finally:
        train_lib.load = orig
    assert f1.equals(f2) and t1.equals(t2)
