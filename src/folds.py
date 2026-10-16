"""Permanent fold files in data/folds/ (columns id, fold) and their validation.

primary   = KFold(shuffle=True, random_state=42) over train rows, exactly the split used by train_lib.Data
            (the test set is a random 30% row sample, decision D001/D002)
secondary = zone_region_id groups assigned to folds at random (seeded), balancing rows per fold
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import load  # noqa: E402

FOLD_DIR = "data/folds"


def primary(ids, k, seed=42):
    fold = np.empty(len(ids), dtype=np.int8)
    for f, (_, va) in enumerate(KFold(k, shuffle=True, random_state=seed).split(ids)):
        fold[va] = f
    return fold


def secondary(groups, k, seed=42):
    """random group order, then each group goes to the fold with the fewest rows so far"""
    uniq, cnt = np.unique(groups, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    sizes, gf = np.zeros(k, dtype=np.int64), {}
    for j in order:
        f = int(np.argmin(sizes))
        gf[uniq[j]] = f
        sizes[f] += cnt[j]
    return np.array([gf[g] for g in groups], dtype=np.int8)


def validate(df, groups=None, k=None):
    assert df["id"].is_unique and df["fold"].notna().all()
    k = k or int(df["fold"].max()) + 1
    sizes = df["fold"].value_counts().sort_index().values
    assert len(sizes) == k and sizes.max() / sizes.min() < 1.05, sizes
    if groups is not None:
        assert (pd.Series(df["fold"].values).groupby(groups).nunique() == 1).all(), "a group is split across folds"
    return sizes


def load_folds(name):
    return pd.read_parquet(f"{FOLD_DIR}/{name}.parquet")


if __name__ == "__main__":
    os.makedirs(FOLD_DIR, exist_ok=True)
    tr, _ = load()
    ids = tr["id"].values
    for name, fold, groups in [("folds_primary_k5_s42", primary(ids, 5), None),
                               ("folds_primary_k10_s42", primary(ids, 10), None),
                               ("folds_secondary_k5_s42", secondary(tr["zone_region_id"].values, 5),
                                tr["zone_region_id"].values)]:
        path = f"{FOLD_DIR}/{name}.parquet"
        df = pd.DataFrame({"id": ids, "fold": fold})
        if os.path.exists(path):  # fold files are permanent: never overwritten, only re-checked
            assert load_folds(name).equals(df), f"{path} differs from the generator"
        else:
            df.to_parquet(path, index=False)
        print(name, validate(df, groups))
