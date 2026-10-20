"""Independent check (verify2_integrity) of the claim: sub_v19vce has gas > 1e6 and EUI > 1e6/A, while train gas < 1e6.
Reads the raw CSV (only 3 columns), the parquet cache, the test floor area and the submission file. No model fitting."""
import json
import numpy as np
import pandas as pd

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
out = {}

# 1. raw train CSV: the bound and the identity, from the original file (not the cache)
raw = pd.read_csv("data/raw/train.csv", usecols=["id", GAS, EUI, EL, "bld_dim_06"])
g, e, a, el = raw[GAS].values, raw[EUI].values, raw["bld_dim_06"].values, raw[EL].values
out["raw_rows"] = int(len(raw))
out["raw_gas_max"] = float(g.max())
out["raw_gas_n_ge_1e6"] = int((g >= 1e6).sum())
out["raw_gas_nan"] = int(np.isnan(g).sum())
out["raw_A_min"] = float(a.min())
out["raw_eui_identity_max_rel"] = float(np.max(np.abs(e * a - g) / g))
out["raw_eui_times_A_max"] = float((e * a).max())
# tail shape of gas: counts per 25k bin from 800k, and a log-linear (exponential) tail fit on [700k, 950k)
edges = np.arange(800_000, 1_100_001, 25_000)
out["raw_gas_tail_counts_25k"] = {f"{int(lo)}-{int(hi)}": int(((g >= lo) & (g < hi)).sum())
                                  for lo, hi in zip(edges[:-1], edges[1:])}
fine = np.arange(950_000, 1_000_001, 10_000)
out["raw_gas_tail_counts_10k_top"] = {f"{int(lo)}-{int(hi)}": int(((g >= lo) & (g < hi)).sum())
                                      for lo, hi in zip(fine[:-1], fine[1:])}
# exponential tail: survival S(x) for x in [700k, 950k]; extrapolate the expected number of rows >= 1e6
xs = np.arange(700_000, 950_001, 25_000)
S = np.array([(g >= x).sum() for x in xs], dtype=float)
b, c0 = np.polyfit(xs, np.log(S), 1)
out["tail_fit_expected_n_ge_1e6_if_untruncated"] = float(np.exp(c0 + b * 1e6))
out["tail_fit_expected_n_ge_975k"] = float(np.exp(c0 + b * 975_000))
out["observed_n_ge_975k"] = int((g >= 975_000).sum())
# electricity: is there a similar hard ceiling?
out["raw_elec_max"] = float(el.max())
eedges = np.arange(300_000, 420_001, 20_000)
out["raw_elec_tail_counts_20k"] = {f"{int(lo)}-{int(hi)}": int(((el >= lo) & (el < hi)).sum())
                                   for lo, hi in zip(eedges[:-1], eedges[1:])}
del raw

# 2. cache parquet equals raw for the gas column (the pipeline reads the cache)
tr = pd.read_parquet("cache/train.parquet", columns=["id", GAS, EUI, "bld_dim_06"])
out["cache_gas_max"] = float(tr[GAS].max())
out["cache_rows"] = int(len(tr))
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06"])
At = te["bld_dim_06"].values
out["test_A_min"] = float(At.min())
out["test_A_nonpos_or_nan"] = int((~(At > 0)).sum())

# 3. the submission file itself
sub = pd.read_csv("submissions/sub_v19vce.csv")
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
out["sub_rows"] = int(len(sub))
out["sub_id_order_matches_sample"] = bool((sub["id"].values == ss["id"].values).all())
out["sub_id_order_matches_test"] = bool((sub["id"].values == te["id"].values).all())
gs, es, els = sub[GAS].values, sub[EUI].values, sub[EL].values
hi_g = np.where(gs > 1e6)[0]
hi_e = np.where(es > 1e6 / At)[0]
out["sub_gas_max"] = float(gs.max())
out["sub_gas_n_gt_1e6"] = int(len(hi_g))
out["sub_gas_gt_1e6_rows"] = [dict(id=sub["id"].values[i], gas=round(float(gs[i]), 1), A=float(At[i]),
                                   eui=round(float(es[i]), 3), eui_times_A=round(float(es[i] * At[i]), 1))
                              for i in hi_g[np.argsort(gs[hi_g])]]
out["sub_eui_n_gt_1e6_over_A"] = int(len(hi_e))
out["sub_eui_gt_bound_rows"] = [dict(id=sub["id"].values[i], eui=round(float(es[i]), 3),
                                     bound=round(float(1e6 / At[i]), 3), excess=round(float(es[i] - 1e6 / At[i]), 3),
                                     gas=round(float(gs[i]), 1)) for i in hi_e[np.argsort(es[hi_e] - 1e6 / At[hi_e])]]
out["sub_gas_n_gt_train_max"] = int((gs > g.max()).sum())
out["sub_rows_gas_or_eui_flag_union"] = int(len(np.union1d(hi_g, hi_e)))
out["sub_rows_gas_and_eui_flag_both"] = int(len(np.intersect1d(hi_g, hi_e)))
out["sub_elec_max"] = float(els.max())
out["sub_elec_n_gt_train_max"] = int((els > el.max()).sum())
out["sub_min"] = {c: float(sub[c].min()) for c in [GAS, EL, EUI]}
# 4. are any flagged rows among the duplicate-copied rows (those carry train values, so must be < 1e6)?
train_gas_set = set(np.round(g, 4).tolist())
out["flagged_rows_equal_to_some_train_gas"] = int(sum(round(float(gs[i]), 4) in train_gas_set for i in hi_g))
json.dump(out, open("exp/audit/verify2_integrity/v1_files.json", "w"), indent=1)
for k, v in out.items():
    if not isinstance(v, (list, dict)):
        print(k, v)
print("gas tail 25k", out["raw_gas_tail_counts_25k"])
print("gas tail 10k top", out["raw_gas_tail_counts_10k_top"])
print("elec tail", out["raw_elec_tail_counts_20k"])
