"""Copy train targets to test rows that duplicate a train row (the same postcode record listed twice).

Match key: socio_demog_005 (near-unique), subzone_id, bld_dim_12, bld_dim_01, bld_dim_06. In train, all 92 groups of
rows sharing this key have identical values for all five targets. 77 test rows match a train row.

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

