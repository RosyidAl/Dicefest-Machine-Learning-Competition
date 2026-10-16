"""Copy train targets to test rows that duplicate a train row (the same postcode record listed twice).

Match key: socio_demog_005 (near-unique), subzone_id, bld_dim_12, bld_dim_01, bld_dim_06. In train, all 92 groups of
rows sharing this key have identical values for all five targets. 77 test rows match a train row.

usage: python src/dup_fix.py <in_tag> <out_tag>   submissions/sub_<in>.csv -> sub_<out>.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import TARGET_COLS, load  # noqa: E402

KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]


def duplicate_targets(tr, te):
    """Train targets for the test rows whose key matches a train row (NaN elsewhere), in test row order."""
    grp = tr.groupby(KEY, sort=False)[TARGET_COLS]
    assert (grp.nunique() == 1).all().all(), "train duplicates disagree on a target"
    ref = grp.first().reset_index()
    out = te[KEY].merge(ref, on=KEY, how="left")
    assert len(out) == len(te)
    return out[TARGET_COLS].values


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    tr, te = load()
    sub = pd.read_csv(f"submissions/sub_{src}.csv", float_precision="round_trip")
    assert (sub["id"].values == te["id"].values).all()
    dup = duplicate_targets(tr, te)
    hit = ~np.isnan(dup[:, 0])
    for j, col in enumerate(TARGET_COLS):
        sub.loc[hit, col] = dup[hit, j]
    print(f"rows copied from train: {int(hit.sum())}")
    sub.to_csv(f"submissions/sub_{dst}.csv", index=False)
    print("wrote", f"submissions/sub_{dst}.csv", sub.shape)
