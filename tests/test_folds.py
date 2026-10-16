import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from folds import load_folds, validate


def test_fold_files():
    tr = pd.read_parquet("data/cache/train.parquet", columns=["id", "zone_region_id"])
    for name in ["folds_primary_k5_s42", "folds_primary_k10_s42"]:
        df = load_folds(name)
        assert (df["id"].values == tr["id"].values).all()
        validate(df)
    sec = load_folds("folds_secondary_k5_s42")
    validate(sec, groups=tr["zone_region_id"].values)


def test_primary_equals_train_lib_split():
    """train_lib.Data uses KFold(k, shuffle=True, random_state=42) on the row order; the files must match it"""
    n = 429303
    for k in (5, 10):
        df = load_folds(f"folds_primary_k{k}_s42")
        for f, (_, va) in enumerate(KFold(k, shuffle=True, random_state=42).split(np.zeros(n))):
            assert (np.flatnonzero(df["fold"].values == f) == va).all()
