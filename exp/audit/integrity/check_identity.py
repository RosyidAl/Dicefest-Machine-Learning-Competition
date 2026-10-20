"""Integrity audit 2: exact-identity columns.

clean ratio: target == round(100*b02/b01, 2)? rounding mode? RMSE of rounded vs unrounded on train; test edge cases.
adoption: constant within subzone on train? submission copies the train value for seen subzones? adoption <= rest?
gas/EUI identity and target ranges (truncation) on train vs submission ranges.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_identity.py
"""
import json
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm", "target_clean_building_ratio",
     "target_electric_heating_adoption"]
SIG = dict(zip(T, [151750.04, 41725.37, 40.65, 12.49, 7.18]))
HV = [f"hvac_dist_{i:02d}" for i in range(1, 11)]
cols = ["id", "subzone_id", "bld_dim_01", "bld_dim_02", "bld_dim_13", "bld_dim_06", "bld_dim_12"] + HV
tr = pd.read_parquet("data/cache/train.parquet", columns=cols + T)
te = pd.read_parquet("data/cache/test.parquet", columns=cols)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
out = {}

# ---------------- clean ratio ----------------
b1, b2 = tr["bld_dim_01"].values.astype(float), tr["bld_dim_02"].values.astype(float)
y = tr[T[3]].values
raw = 100 * b2 / b1
r_np = np.round(raw, 2)
d = {}
d["train_b01_le0_or_nan"] = int(((b1 <= 0) | np.isnan(b1)).sum())
d["train_b02_nan"] = int(np.isnan(b2).sum())
d["train_target_nan"] = int(np.isnan(y).sum())
d["train_mismatch_np_round"] = int((r_np != y).sum())
d["train_max_abs_diff_np_round"] = float(np.nanmax(np.abs(r_np - y)))
d["train_rmse_unrounded_vs_target"] = float(np.sqrt(np.mean((raw - y) ** 2)))
d["total_score_cost_if_unrounded"] = d["train_rmse_unrounded_vs_target"] / SIG[T[3]] / 5
# alternative formulas
d["train_mismatch_b13"] = int((np.round(100 * (b1 - tr["bld_dim_13"].values) / b1, 2) != y).sum())
# decimal rounding modes on the exact decimal value of the float raw (python round on float = half-even on binary)
half = np.abs(raw * 100 - np.floor(raw * 100) - 0.5) < 1e-6
d["train_near_half_ties"] = int(half.sum())
if half.any():
    idx = np.where(half)[0]
    up = np.array([float(Decimal(repr(float(raw[i]))).quantize(Decimal("0.01"), ROUND_HALF_UP)) for i in idx])
    ev = np.array([float(Decimal(repr(float(raw[i]))).quantize(Decimal("0.01"), ROUND_HALF_EVEN)) for i in idx])
    d["ties_np_round_correct"] = int((r_np[idx] == y[idx]).sum())
    d["ties_half_up_correct"] = int((up == y[idx]).sum())
    d["ties_half_even_decimal_correct"] = int((ev == y[idx]).sum())
    exact_frac = np.array([Decimal(int(b2[i]) * 100) / Decimal(int(b1[i])) for i in idx])
    up_exact = np.array([float(x.quantize(Decimal("0.01"), ROUND_HALF_UP)) for x in exact_frac])
    d["ties_exact_rational_half_up_correct"] = int((up_exact == y[idx]).sum())
d["train_b02_gt_b01"] = int((b2 > b1).sum())
d["train_b01_integer"] = bool(np.all(b1 == np.round(b1)))
# test
t1, t2 = te["bld_dim_01"].values.astype(float), te["bld_dim_02"].values.astype(float)
d["test_b01_le0_or_nan"] = int(((t1 <= 0) | np.isnan(t1)).sum())
d["test_b02_nan"] = int(np.isnan(t2).sum())
d["test_b02_gt_b01"] = int((t2 > t1).sum())
traw = 100 * t2 / t1
d["test_sub_equals_np_round"] = int((np.round(traw, 2) == sub[T[3]].values).sum())
d["test_sub_mismatch_np_round"] = int((np.round(traw, 2) != sub[T[3]].values).sum())
thalf = np.abs(traw * 100 - np.floor(traw * 100) - 0.5) < 1e-6
d["test_near_half_ties"] = int(thalf.sum())
if thalf.any():
    idx = np.where(thalf)[0]
    up_exact = np.array([float((Decimal(int(t2[i]) * 100) / Decimal(int(t1[i]))).quantize(Decimal("0.01"),
                                                                                         ROUND_HALF_UP)) for i in idx])
    d["test_ties_np_vs_exact_half_up_differ"] = int((up_exact != np.round(traw[idx], 2)).sum())
out["clean_ratio"] = d

# ---------------- adoption ----------------
a = {}
ya = tr[T[4]].values
g = tr.groupby("subzone_id")[T[4]]
nun = g.nunique()
a["train_subzones"] = int(len(nun))
a["train_subzones_nonconstant"] = int((nun > 1).sum())
rng = (g.max() - g.min())
a["train_subzone_max_range"] = float(rng.max())
rest = 100 - tr[HV].sum(axis=1).values
a["train_adoption_gt_rest_plus_1e-6"] = int((ya > rest + 1e-6).sum())
a["train_adoption_lt0"] = int((ya < 0).sum())
# hvac constant within subzone?
a["train_hvac_nonconstant_subzones"] = int((tr.groupby("subzone_id")[HV].nunique() > 1).any(axis=1).sum())
ref = g.first()
seen = te["subzone_id"].isin(ref.index).values
a["test_seen_rows"] = int(seen.sum())
a["test_unseen_rows"] = int((~seen).sum())
copy = te["subzone_id"].map(ref).values
sa = sub[T[4]].values
a["test_seen_sub_equals_train_value"] = int((sa[seen] == copy[seen]).sum())
a["test_seen_sub_max_abs_diff"] = float(np.max(np.abs(sa[seen] - copy[seen])))
trest = 100 - te[HV].sum(axis=1).values
a["test_sub_gt_rest_plus_1e-9"] = int((sa > trest + 1e-9).sum())
a["test_sub_gt_rest_max_excess"] = float(np.max(sa - trest))
a["test_rest_min"] = float(trest.min())
# unseen subzones: is the prediction constant within each unseen subzone?
un = te.loc[~seen, ["subzone_id"]].copy()
un["p"] = sa[~seen]
gu = un.groupby("subzone_id")["p"]
a["test_unseen_subzones"] = int(gu.ngroups)
a["test_unseen_subzones_multi_row"] = int((gu.size() > 1).sum())
a["test_unseen_rows_in_multi_row_subzones"] = int(gu.size()[gu.size() > 1].sum())
a["test_unseen_subzones_nonconstant_pred"] = int((gu.nunique() > 1).sum())
a["test_unseen_within_subzone_pred_sd_rms"] = float(np.sqrt(np.mean((un["p"] - gu.transform("mean")) ** 2)))
# hvac constant within unseen test subzones?
a["test_hvac_nonconstant_subzones"] = int((te.groupby("subzone_id")[HV].nunique() > 1).any(axis=1).sum())
out["adoption"] = a

# ---------------- gas / EUI identity and ranges ----------------
e = {}
A = tr["bld_dim_06"].values.astype(float)
e["train_eui_identity_max_rel"] = float(np.max(np.abs(tr[T[0]].values / A - tr[T[2]].values) / tr[T[2]].values))
for c in T[:3]:
    e[c] = dict(train_min=float(tr[c].min()), train_max=float(tr[c].max()), sub_min=float(sub[c].min()),
                sub_max=float(sub[c].max()), sub_n_gt_train_max=int((sub[c] > tr[c].max()).sum()),
                sub_n_lt_train_min=int((sub[c] < tr[c].min()).sum()),
                train_n_ge_1e6=int((tr[c] >= 1e6).sum()))
e["train_gas_top5"] = sorted(tr[T[0]].values)[-5:]
e["train_elec_top5"] = sorted(tr[T[1]].values)[-5:]
e["train_gas_bottom5"] = sorted(tr[T[0]].values)[:5]
e["train_elec_bottom5"] = sorted(tr[T[1]].values)[:5]
e["train_eui_top5"] = sorted(tr[T[2]].values)[-5:]
e["train_eui_bottom5"] = sorted(tr[T[2]].values)[:5]
out["gas_eui"] = e
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("exp/audit/integrity/check_identity.json", "w"), indent=1, default=float)
