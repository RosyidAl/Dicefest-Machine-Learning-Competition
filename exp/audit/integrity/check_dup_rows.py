"""Integrity audit 6: the 77 copied rows in the submitted file, test-test duplicates, adoption rows with rest == 0.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_dup_rows.py
"""
import json

import numpy as np
import pandas as pd

T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm", "target_clean_building_ratio",
     "target_electric_heating_adoption"]
SIG = [151750.04, 41725.37, 40.65, 12.49, 7.18]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
HV = [f"hvac_dist_{i:02d}" for i in range(1, 11)]
tr = pd.read_parquet("data/cache/train.parquet", columns=["id"] + KEY + T)
te = pd.read_parquet("data/cache/test.parquet", columns=["id"] + KEY + HV)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
prev = pd.read_csv("submissions/sub_v19vc.csv", float_precision="round_trip")
out = {}
ref = tr.groupby(KEY, sort=False)[T].first().reset_index()
m = te[KEY].merge(ref, on=KEY, how="left")
hit = ~m[T[0]].isna().values
out["dup_rows"] = int(hit.sum())
out["dup_rows_sub_equals_train_all_targets"] = int((sub.loc[hit, T].values == m.loc[hit, T].values).all(axis=1).sum())
# column-level diff vs previous submission (sub_v19vc): which columns changed
out["diff_vs_v19vc"] = {c: int((sub[c].values != prev[c].values).sum()) for c in T}
# test-test duplicates on KEY (not matching train): prediction spread within group
tt = te[KEY].copy()
for c in T[:3]:
    tt[c] = sub[c].values
tt["hit"] = hit
gs = tt.groupby(KEY, sort=False)
size = gs[T[0]].transform("size").values
multi = (size > 1) & ~hit
out["test_test_dup_rows_not_in_train"] = int(multi.sum())
gain = {}
for j, c in enumerate(T[:3]):
    mean = gs[c].transform("mean").values
    # SSE reduction from averaging within group is guaranteed >= sum((p - mean)^2) ... exact identity:
    # sum_i (p_i - y)^2 - sum_i (mean - y)^2 = sum_i (p_i - mean)^2 when all rows share y
    red = np.sum((tt[c].values[multi] - mean[multi]) ** 2)
    rmse_now = 0.245 * SIG[j] if j == 0 else (0.315 * SIG[j] if j == 1 else 0.649 * SIG[j])
    mse_now = rmse_now ** 2
    d_rmse = np.sqrt(mse_now - red / len(te)) - rmse_now
    gain[c] = dict(sse_reduction=float(red), approx_delta_nrmse=float(d_rmse / SIG[j]),
                   approx_delta_total=float(d_rmse / SIG[j] / 5))
out["test_test_avg_gain"] = gain
# adoption: rows where rest == 0 must have adoption 0
rest = 100 - te[HV].sum(axis=1).values
z = np.abs(rest) < 1e-9
out["test_rest_zero_rows"] = int(z.sum())
out["test_rest_zero_sub_adoption_max"] = float(sub.loc[z, T[4]].max()) if z.any() else None
out["test_rest_zero_sub_adoption_nonzero"] = int((sub.loc[z, T[4]].values > 0).sum())
seen = te["subzone_id"].isin(tr["subzone_id"]).values
out["test_rest_zero_unseen_rows"] = int((z & ~seen).sum())
out["test_rest_zero_unseen_sub_adoption_max"] = float(sub.loc[z & ~seen, T[4]].max()) if (z & ~seen).any() else None
# unseen rows: adoption prediction exceeding rest
ex = sub[T[4]].values - rest
out["test_unseen_excess_over_rest_max"] = float(ex[~seen].max())
out["test_unseen_rows_excess_gt_0"] = int((ex[~seen] > 1e-12).sum())
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_dup_rows.json", "w"), indent=1)
