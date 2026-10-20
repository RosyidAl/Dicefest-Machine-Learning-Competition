"""Adoption lens, check 1: verify the adoption column of the submitted file end to end.

- adoption constant within subzone in train (exact)
- seen test subzones: submitted value == train value (exact)
- unseen test subzones: constant within subzone, within [0, rest], no NaN, 0..100
- submitted column == exp/pred/adopt_v2.npy (after clip), id order == sample submission
- subzone-level census columns: identical across all train AND test rows of a subzone?
usage: .venv/bin/python exp/audit/adoption/a1_verify_sub.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import HVAC  # noqa: E402

AD = "target_electric_heating_adoption"
OUT = "exp/audit/adoption"

tr = pd.read_parquet("cache/train.parquet")
te = pd.read_parquet("cache/test.parquet")
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
smp = pd.read_parquet("cache/sample.parquet")
res = {}

res["id_order_equals_sample"] = bool((sub["id"].values == smp["id"].values).all())
res["id_order_equals_test"] = bool((sub["id"].values == te["id"].values).all())
res["n_rows"] = int(len(sub))
a = sub[AD].values
res["adopt_nan"] = int(np.isnan(a).sum())
res["adopt_min"], res["adopt_max"] = float(a.min()), float(a.max())

# 1. constancy in train
g = tr.groupby("subzone_id")[AD]
rng = (g.max() - g.min())
res["train_subzones"] = int(len(rng))
res["train_subzones_nonconstant_adoption"] = int((rng > 0).sum())
trval = g.first()

# 2. seen rows
seen = te["subzone_id"].isin(trval.index).values
res["test_rows_seen"] = int(seen.sum())
res["test_rows_unseen"] = int((~seen).sum())
res["unseen_row_frac"] = float((~seen).mean())
copy = te.loc[seen, "subzone_id"].map(trval).values
d = a[seen] - copy
res["seen_rows_exact_equal"] = int((d == 0).sum())
res["seen_rows_max_absdiff"] = float(np.abs(d).max())

# 3. unseen rows: constant within subzone
us = pd.DataFrame({"s": te.loc[~seen, "subzone_id"].values, "a": a[~seen]})
gu = us.groupby("s")["a"]
res["unseen_subzones"] = int(gu.ngroups)
res["unseen_subzones_nonconstant_pred"] = int((gu.max() - gu.min() > 0).sum())

# 4. rest / K bounds for unseen
rest_te = 100 - te[HVAC].sum(axis=1).values
res["unseen_pred_gt_rest"] = int((a[~seen] > rest_te[~seen] + 1e-9).sum())
res["unseen_pred_lt_0"] = int((a[~seen] < 0).sum())
res["unseen_rest_zero_rows"] = int((np.abs(rest_te[~seen]) < 1e-9).sum())
res["unseen_rest_zero_pred_nonzero"] = int(((np.abs(rest_te[~seen]) < 1e-9) & (a[~seen] != 0)).sum())
# train: adoption <= rest always?
rest_tr = 100 - tr[HVAC].sum(axis=1).values
res["train_adopt_gt_rest_rows"] = int((tr[AD].values > rest_tr + 1e-6).sum())
res["train_rest_zero_rows"] = int((np.abs(rest_tr) < 1e-9).sum())
res["train_rest_zero_adopt_nonzero"] = int(((np.abs(rest_tr) < 1e-9) & (tr[AD].values != 0)).sum())

# 5. equals adopt_v2.npy
p2 = np.clip(np.load("exp/pred/adopt_v2.npy"), 0, 100)
res["equals_adopt_v2_npy"] = int((p2 == a).sum())
res["max_absdiff_adopt_v2_npy"] = float(np.abs(p2 - a).max())

# 6. census constancy within subzone (train only, and train+test)
cols = [c for c in te.columns if c not in ("id", "subzone_id", "zone_region_id")]
al = pd.concat([tr[["subzone_id"] + cols], te[["subzone_id"] + cols]], ignore_index=True)
nun_tr = tr.groupby("subzone_id")[cols].nunique(dropna=False).max()
nun_al = al.groupby("subzone_id")[cols].nunique(dropna=False).max()
const_tr = [c for c in cols if nun_tr[c] <= 1]
broken = [c for c in const_tr if nun_al[c] > 1]
res["n_cols_constant_within_subzone_train"] = len(const_tr)
res["cols_constant_train_but_not_train_test"] = broken
res["hvac_constant_train"] = bool(all(nun_tr[h] <= 1 for h in HVAC))
res["hvac_constant_train_test"] = bool(all(nun_al[h] <= 1 for h in HVAC))
groups = {}
for c in const_tr:
    p = c.rsplit("_", 1)[0]
    groups.setdefault(p, []).append(c)
res["const_cols_by_prefix"] = {k: len(v) for k, v in groups.items()}
nonconst = [c for c in cols if nun_tr[c] > 1]
res["nonconst_socio_cols"] = [c for c in nonconst if c.startswith("socio")]
res["nonconst_env_cols"] = [c for c in nonconst if c.startswith("env")]
# also the zone: is subzone nested within zone?
res["subzones_in_multiple_zones"] = int((pd.concat([tr[["subzone_id", "zone_region_id"]], te[["subzone_id", "zone_region_id"]]])
                                         .groupby("subzone_id")["zone_region_id"].nunique() > 1).sum())
json.dump({"const_cols": const_tr}, open(f"{OUT}/const_cols.json", "w"))
json.dump(res, open(f"{OUT}/a1_verify_sub.json", "w"), indent=1)
for k, v in res.items():
    print(k, v)
