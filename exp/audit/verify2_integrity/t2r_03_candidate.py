"""verify2_integrity t2r_03: check the integrity candidate file on the actual submissions.
  1. sub_v19vce_trunc2.csv vs submissions/sub_v19vce.csv: ids, untouched columns, gas / EUI changes, duplicate rows.
  2. own re-implementation of the claimed rule (parameters from make_trunc_candidate2.json, formula from the claim
     text) applied to sub_v19vce -> numeric agreement with the candidate and sha256 of a default-format CSV.
  3. overlap with BEST_CURRENT (sub_v19nca2pp, already contains the D015 S3 soft cap): how the high-gas rows look there.
Reads only light column subsets of cache/*.parquet. Writes only under exp/audit/verify2_integrity/.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/t2r_03_candidate.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

D = "exp/audit/verify2_integrity"
GAS, EL, EUI, CL, AD = ("target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm",
                        "target_clean_building_ratio", "target_electric_heating_adoption")
T = [GAS, EL, EUI, CL, AD]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
rd = lambda f: pd.read_csv(f, float_precision="round_trip")  # noqa: E731
base = rd("submissions/sub_v19vce.csv")
cand = rd("exp/audit/integrity/sub_v19vce_trunc2.csv")
best = rd("submissions/BEST_CURRENT.csv")
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
tr = pd.read_parquet("cache/train.parquet", columns=KEY + T)
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
out = {}
out["ids_equal"] = bool((base["id"].values == cand["id"].values).all() and (cand["id"].values == ss["id"].values).all()
                        and (te["id"].values == ss["id"].values).all() and (best["id"].values == ss["id"].values).all())
out["untouched_identical"] = {c: bool(np.array_equal(base[c].values, cand[c].values)) for c in (EL, CL, AD)}
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
# duplicates: test rows whose KEY equals a train row (own merge)
ref = tr.groupby(KEY, sort=False)[T].first().reset_index()
dup = te[KEY].merge(ref, on=KEY, how="left")[T].values
isdup = ~np.isnan(dup[:, 0])
out["n_dup"] = int(isdup.sum())
out["dup_rows_untouched_in_candidate"] = bool(np.array_equal(cand.loc[isdup, T].values, base.loc[isdup, T].values))
out["dup_rows_equal_train_targets_in_base"] = bool(np.array_equal(base.loc[isdup, T].values, dup[isdup]))
for c, nm in ((GAS, "gas"), (EUI, "eui")):
    p0, p1 = base[c].values, cand[c].values
    d = p1 - p0
    out[nm] = dict(rows_changed_gt_1pct=int((np.abs(d) > 0.01 * np.abs(p0)).sum()), max_increase=float(d.max()),
                   min_change=float(d.min()), argmin_before=float(p0[np.argmin(d)]), argmin_after=float(p1[np.argmin(d)]),
                   max_before_gas_scale=float((p0 * (At if nm == "eui" else 1)).max()),
                   max_after_gas_scale=float((p1 * (At if nm == "eui" else 1)).max()),
                   n_before_ge_1e6_gas_scale=int((p0 * (At if nm == "eui" else 1) >= 1e6).sum()),
                   rms_change_over_sigma=float(np.sqrt(np.mean(d ** 2)) / (151750.04 if nm == "gas" else 40.65)))

# own implementation of the claimed rule
prm = json.load(open("exp/audit/integrity/make_trunc_candidate2.json"))["params"]
a0, b0, k, cg, ce = prm["a0"], prm["b0"], prm["k"], prm["c_gas"], prm["c_eui"]


def tadj(pg):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / Mt, 1e-6)), 1.0)
    a = (1e6 - pg) / s
    return -s * np.exp(-0.5 * a * a - 0.5 * np.log(2 * np.pi) - log_ndtr(a))


mine = base.copy()
g, e = base[GAS].values.astype(float), base[EUI].values.astype(float)
gn = np.clip(g + cg * tadj(g), 0, None)
en = np.clip((e * At + ce * tadj(e * At)) / At, 0, None)
mine[GAS] = np.where(isdup, g, gn)
mine[EUI] = np.where(isdup, e, en)
out["own_vs_candidate"] = {nm: dict(max_abs_diff=float(np.max(np.abs(mine[c].values - cand[c].values))),
                                    max_rel_diff=float(np.max(np.abs(mine[c].values - cand[c].values) /
                                                              np.maximum(np.abs(cand[c].values), 1e-12))),
                                    n_exact=int((mine[c].values == cand[c].values).sum()))
                           for c, nm in ((GAS, "gas"), (EUI, "eui"))}
f = f"{D}/t2r_03_own_trunc2.csv"
mine.to_csv(f, index=False)
out["own_csv_sha256"] = hashlib.sha256(open(f, "rb").read()).hexdigest()
out["candidate_sha256"] = hashlib.sha256(open("exp/audit/integrity/sub_v19vce_trunc2.csv", "rb").read()).hexdigest()

# overlap with BEST_CURRENT (already has the D015 S3 soft cap): rows the claim moves most
idx = np.argsort(cand[GAS].values - base[GAS].values)[:15]
out["top15_gas_moves"] = [dict(v19vce=round(float(base[GAS].values[i])), trunc2=round(float(cand[GAS].values[i])),
                               best_current=round(float(best[GAS].values[i])), M=float(Mt[i])) for i in idx]
hi = base[GAS].values > 9e5
out["best_current_high_rows"] = dict(n_v19vce_gt_900k=int(hi.sum()),
                                     mean_trunc2_minus_v19vce=float((cand[GAS].values - base[GAS].values)[hi].mean()),
                                     mean_best_minus_v19vce=float((best[GAS].values - base[GAS].values)[hi].mean()),
                                     max_best_gas=float(best[GAS].max()), max_v19vce_gas=float(base[GAS].max()),
                                     max_trunc2_gas=float(cand[GAS].max()))
print(json.dumps(out, indent=1))
json.dump(out, open(f"{D}/t2r_03_candidate.json", "w"), indent=1)
