"""Adoption lens (round 2), check b1: independent end-to-end verification of the adoption column.

Reads only the columns it needs (pyarrow column subsets) to stay well under 3 GB.
Checks on submissions/sub_v19vce.csv (and that BEST_CURRENT / sub_v19nca2pp carry the same adoption column):
 1. id order == sample submission == test
 2. no NaN/inf, 0 <= a <= 100
 3. train: adoption constant within subzone
 4. seen test rows: value == train value of the subzone (exact)
 5. unseen test rows: constant within subzone; 0 <= a <= rest (float64 rest); rest == 0 rows
 6. every feature column that is constant within subzone in train is also constant over train + test rows
 7. household-count lattice in train: N from hvac shares, adoption * N / 100 integral? (N ambiguity)
 8. submitted column == clip(exp/pred/adopt_v2.npy)
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/adoption/r2/b1_verify.py
"""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "src")
from adopt_v2 import UNSEEN_MIX, household_count  # noqa: E402
from common import HVAC  # noqa: E402

AD = "target_electric_heating_adoption"
OUT = "exp/audit/adoption/r2"
res = {}

tr = pd.read_parquet("cache/train.parquet", columns=["id", "subzone_id", AD] + HVAC)
te = pd.read_parquet("cache/test.parquet", columns=["id", "subzone_id"] + HVAC)
smp = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
subs = {}
for name in ("sub_v19vce", "sub_v19nca2pp", "BEST_CURRENT"):
    subs[name] = pd.read_csv(f"submissions/{name}.csv", usecols=["id", AD], float_precision="round_trip")
sub = subs["sub_v19vce"]
a = sub[AD].values

# 1. ids
res["id_eq_sample_raw"] = bool((sub.id.values == smp.id.values).all())
res["id_eq_test"] = bool((sub.id.values == te.id.values).all())
for name, s in subs.items():
    res[f"adopt_identical_{name}"] = bool(np.array_equal(s[AD].values, a)) and bool((s.id.values == sub.id.values).all())

# 2. range
res["n_rows"] = len(a)
res["nonfinite"] = int((~np.isfinite(a)).sum())
res["min"], res["max"] = float(a.min()), float(a.max())

# 3. train constancy
g = tr.groupby("subzone_id")[AD]
res["train_subzones"] = int(g.ngroups)
res["train_subzones_nonconstant"] = int(((g.max() - g.min()) != 0).sum())
trval = g.first()

# 4. seen rows
seen = te.subzone_id.isin(trval.index).values
res["test_rows_seen"], res["test_rows_unseen"] = int(seen.sum()), int((~seen).sum())
d = a[seen] - te.subzone_id[seen].map(trval).values
res["seen_exact_equal"] = int((d == 0).sum())
res["seen_max_absdiff"] = float(np.abs(d).max())

# 5. unseen rows
rest_te = 100.0 - te[HVAC].sum(axis=1).values  # float64
us = pd.DataFrame({"s": te.subzone_id.values[~seen], "a": a[~seen], "rest": rest_te[~seen]})
gu = us.groupby("s")
res["unseen_subzones"] = int(gu.ngroups)
res["unseen_subzones_nonconstant_pred"] = int(((gu.a.max() - gu.a.min()) != 0).sum())
nrow_u = gu.size()
mix = nrow_u.clip(upper=6).map(lambda k: k).value_counts().sort_index()
res["unseen_row_mix"] = {int(k): int(v * k if k < 6 else 0) for k, v in mix.items()}
res["unseen_row_mix_6plus_rows"] = int(nrow_u[nrow_u >= 6].sum())
res["UNSEEN_MIX_in_code"] = UNSEEN_MIX
exc = us.a.values - us.rest.values
res["unseen_pred_gt_rest_rows"] = int((exc > 0).sum())
res["unseen_pred_gt_rest_max_excess"] = float(exc.max())
rz = np.abs(us.rest.values) < 1e-9
res["unseen_rest0_rows"] = int(rz.sum())
res["unseen_rest0_pred_nonzero_rows"] = int((rz & (us.a.values != 0)).sum())
res["unseen_rest0_pred_max"] = float(us.a.values[rz].max()) if rz.any() else 0.0
res["unseen_pred_lt0"] = int((us.a.values < 0).sum())
rest_tr = 100.0 - tr[HVAC].sum(axis=1).values
res["train_ad_gt_rest_rows"] = int((tr[AD].values > rest_tr + 1e-9).sum())
res["train_rest0_rows"] = int((np.abs(rest_tr) < 1e-9).sum())
res["train_rest0_ad_nonzero"] = int(((np.abs(rest_tr) < 1e-9) & (tr[AD].values != 0)).sum())
res["train_rest_min"] = float(rest_tr.min())
res["test_rest_min"] = float(rest_te.min())

# 8. equals adopt_v2.npy
p2 = np.clip(np.load("exp/pred/adopt_v2.npy"), 0, 100)
res["eq_adopt_v2_npy"] = int((p2 == a).sum())

# 7. lattice: N from hvac (float64), adoption count integral?
S = tr.groupby("subzone_id")[HVAC + [AD]].first()
N = household_count(S[HVAC].values)
ad = S[AD].values
res["N_not_found"] = int((N == 0).sum())
x = ad * N / 100
off = np.abs(x - np.round(x)) > 1e-6 * np.maximum(1, N / 100)
res["subzones_adcount_not_integral"] = int(off.sum())
# smallest multiple m of N making hvac AND adoption integral
mult = np.zeros(len(S), int)
V = np.c_[S[HVAC].values, ad] / 100
for m in range(1, 21):
    Nm = N * m
    ok = (mult == 0) & (np.abs(V * Nm[:, None] - np.round(V * Nm[:, None])) < 2e-4 * np.maximum(1, Nm / 100)[:, None]).all(1)
    mult[ok] = m
res["N_multiplier_counts"] = {int(k): int(v) for k, v in pd.Series(mult).value_counts().sort_index().items()}
# among off subzones, what is the true N then (N * mult) and how big is N_hat
res["off_N_hat_median"] = float(np.median(N[off])) if off.any() else None
res["off_N_true_median"] = float(np.median((N * mult)[off])) if off.any() else None

# 6. census constancy over train + test, column by column
names_te = [c for c in pq.read_schema("cache/test.parquet").names if c not in ("id", "subzone_id")]
codes, uniq = pd.factorize(pd.concat([tr.subzone_id, te.subzone_id], ignore_index=True))
ntr = len(tr)
codes_tr = codes[:ntr]
const_tr, broken, const_all = [], [], []
for c in names_te:
    v = pd.concat([pd.read_parquet("cache/train.parquet", columns=[c])[c],
                   pd.read_parquet("cache/test.parquet", columns=[c])[c]], ignore_index=True)
    n_all = v.groupby(codes).nunique(dropna=False).max()
    n_tr = v.iloc[:ntr].groupby(codes_tr).nunique(dropna=False).max()
    if n_tr <= 1:
        const_tr.append(c)
        if n_all > 1:
            broken.append(c)
    if n_all <= 1:
        const_all.append(c)
res["n_feature_cols"] = len(names_te)
res["n_const_within_subzone_train"] = len(const_tr)
res["n_const_within_subzone_train_test"] = len(const_all)
res["const_train_not_train_test"] = broken
res["hvac_all_const_train_test"] = all(h in const_all for h in HVAC)
res["const_train_test_cols"] = const_all
json.dump(res, open(f"{OUT}/b1_verify.json", "w"), indent=1)
for k, v in res.items():
    if k != "const_train_test_cols":
        print(k, v)
