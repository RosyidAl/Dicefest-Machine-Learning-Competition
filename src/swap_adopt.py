"""Replace the adoption column of an existing submission with another adoption prediction.
Rows of subzones seen in train must be unchanged (both versions copy the train value there).

usage: python src/swap_adopt.py <in_tag> <out_tag> <adopt_pred.npy>   submissions/sub_<in>.csv -> sub_<out>.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import TARGET_COLS, load  # noqa: E402

if __name__ == "__main__":
    src, dst, path = sys.argv[1:4]
    tr, te = load()
    sub = pd.read_csv(f"submissions/sub_{src}.csv", float_precision="round_trip")
    assert (sub["id"].values == te["id"].values).all()
    new = np.clip(np.load(path), 0, 100)
    col = TARGET_COLS[4]
    seen = te["subzone_id"].isin(set(tr["subzone_id"])).values
    assert np.allclose(sub[col].values[seen], new[seen], rtol=0, atol=1e-9), "seen-subzone adoption differs"
    print(f"unseen rows {int((~seen).sum())}, mean |change| {np.abs(sub[col].values[~seen] - new[~seen]).mean():.4f}")
    sub[col] = new
    sub.to_csv(f"submissions/sub_{dst}.csv", index=False)
    print("wrote", f"submissions/sub_{dst}.csv", sub.shape)
