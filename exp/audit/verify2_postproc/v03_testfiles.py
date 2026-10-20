"""Check the small-meter shift (S1) on the actual test files and estimate its expected test effect honestly.

1. Test M distribution, duplicate rows (production dup_fix.duplicate_targets), id order.
2. S1-only candidate from submissions/sub_v19vce.csv with the full-train VC deltas (v02), duplicates and M == 1 untouched.
3. Compare with the prior auditor's candidate exp/audit/postproc/sub_v19vce_pp.csv on the S1 rows.
4. Expected test gain with the sampling-error discount, and the spread of the realised gain on the public (30%) and
   private (70%) parts, from per-row train-OOF error changes.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "final/src")
from dup_fix import KEY, duplicate_targets  # noqa: E402

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
CL, AD = "target_clean_building_ratio", "target_electric_heating_adoption"
TC = [GAS, EL, EUI, CL, AD]
COL = {"gas": GAS, "eui": EUI, "elec": EL}
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
NRMSE_H = {"gas": 0.24514, "eui": 0.64859, "elec": 0.31462}  # honest corrected level on H (chain_H)
out = {}

tr = pd.read_parquet("cache/train.parquet", columns=sorted(set(KEY + TC)))
te = pd.read_parquet("cache/test.parquet", columns=sorted(set(KEY + ["id", "bld_dim_06"])))
dup = duplicate_targets(tr, te)
isdup = ~np.isnan(dup[:, 0])
del tr
Mt = te["bld_dim_12"].values.astype(float)
sub = pd.read_csv("submissions/sub_v19vce.csv")
assert (sub["id"].values == te["id"].values).all()
out["n_test"] = len(sub)
out["dup_rows"] = int(isdup.sum())
out["dup_rows_already_copied"] = bool(np.allclose(sub.loc[isdup, TC].values, dup[isdup], rtol=0, atol=1e-9))
vals, cnt = np.unique(Mt[Mt <= 5], return_counts=True)
out["test_M_counts_le5"] = {str(int(v)): int(c) for v, c in zip(vals, cnt)}
G = {"G23": (Mt >= 2) & (Mt <= 3), "G4": Mt == 4}
out["dup_in_groups"] = {g: int((m & isdup).sum()) for g, m in G.items()}
print(json.dumps(out))

v2 = json.load(open("exp/audit/verify2_postproc/v02_fulloof.json"))
D = v2["deltas_all_rows"]
SE = {m: {g: v2["bias"][m][g]["se"] for g in G} for m in COL}

# 2. S1-only candidate
new = sub.copy()
for m, c in COL.items():
    p = sub[c].values.astype(float).copy()
    for g, msk in G.items():
        sel = msk & ~isdup
        p[sel] = p[sel] * (1 + D[m][g])
    new[c] = np.clip(p, 0, None)
assert (new[[CL, AD]].values == sub[[CL, AD]].values).all()
assert np.array_equal(new.loc[isdup, TC].values, sub.loc[isdup, TC].values)
assert np.isfinite(new[TC].values).all() and (new[[GAS, EL, EUI]].values >= 0).all()
fn = "exp/audit/verify2_postproc/sub_v19vce_s1only.csv"
new.to_csv(fn, index=False)
out["s1only_csv"] = fn
out["s1only_sha256"] = hashlib.sha256(open(fn, "rb").read()).hexdigest()
out["rows_changed"] = {m: int((new[c].values != sub[c].values).sum()) for m, c in COL.items()}

# 3. prior candidate vs S1-only on S1 rows
pp = pd.read_csv("exp/audit/postproc/sub_v19vce_pp.csv")
assert (pp["id"].values == sub["id"].values).all()
cmp_ = {}
for m, c in COL.items():
    for g, msk in G.items():
        sel = msk & ~isdup
        ratio = pp[c].values[sel] / sub[c].values[sel]
        cmp_[f"{m}_{g}"] = dict(expected=1 + D[m][g], ratio_median=float(np.median(ratio)), ratio_min=float(ratio.min()),
                                ratio_max=float(ratio.max()),
                                n_exact=int(np.isclose(ratio, 1 + D[m][g], rtol=1e-9, atol=0).sum()), n=int(sel.sum()))
    oth = ~(G["G23"] | G["G4"])
    cmp_[f"{m}_other_rows_changed_in_pp"] = int((pp[c].values[oth] != sub[c].values[oth]).sum())
cmp_["M1_row_changed_in_pp"] = {m: bool((pp[c].values[Mt == 1] != sub[c].values[Mt == 1]).any()) for m, c in COL.items()}
out["prior_candidate_vs_s1"] = cmp_

# 4. expected test effect. Per metric: E[dSSE] = sum p^2 (d^2 - 2 d delta); with delta ~ d_hat and sampling se:
#    E[d_hat^2 - 2 d_hat delta] = se^2 - delta^2 ~ 2 se^2 - d_hat^2 (conservative: delta^2 ~ d_hat^2 - se^2).
pf = np.load("exp/audit/verify2_postproc/v02_pf.npz")
Mtr = pf["M"]
ytr = pd.read_parquet("cache/train.parquet", columns=[GAS, EL, EUI])
ytr = {"gas": ytr[GAS].values, "eui": ytr[EUI].values, "elec": ytr[EL].values}
Gtr = {"G23": (Mtr >= 2) & (Mtr <= 3), "G4": Mtr == 4}
N = len(sub)
exp_tot, exp_tot_naive = 0.0, 0.0
detail = {}
for m, c in COL.items():
    sse0 = N * (NRMSE_H[m] * SIG[m]) ** 2
    dsse, dsse_naive = 0.0, 0.0
    for g in G:
        sel = G[g] & ~isdup
        p2 = np.sum(sub[c].values[sel] ** 2)
        d, se = D[m][g], SE[m][g]
        dsse += p2 * (2 * se ** 2 - d ** 2)
        dsse_naive += -p2 * d ** 2
    d_nrmse = (np.sqrt((sse0 + dsse) / N) - np.sqrt(sse0 / N)) / SIG[m]
    d_nrmse_naive = (np.sqrt((sse0 + dsse_naive) / N) - np.sqrt(sse0 / N)) / SIG[m]
    detail[m] = dict(expected_dnrmse=float(d_nrmse), naive_dnrmse=float(d_nrmse_naive))
    exp_tot += d_nrmse / 5
    exp_tot_naive += d_nrmse_naive / 5
# spread of the realised TOTAL change on a random part (fraction f of the test rows): per train-OOF row of a group,
# combined contribution c_i = sum_m (d^2 p^2 - 2 d p r) / (2 NRMSE_m sigma_m^2 * 5); metrics are correlated per row,
# so they are combined before taking the variance.
sd_part = {}
for part, f in (("public", 0.3), ("private", 0.7)):
    Np = f * N
    var = 0.0
    for g in G:
        n_part = f * int((G[g] & ~isdup).sum())
        ci = 0.0
        for m in COL:
            pt, rt = pf[m][Gtr[g]], ytr[m][Gtr[g]] - pf[m][Gtr[g]]
            d = D[m][g]
            ci = ci + (d ** 2 * pt ** 2 - 2 * d * pt * rt) / (2 * NRMSE_H[m] * SIG[m] ** 2 * 5)
        var += n_part * ci.var() / Np ** 2
    sd_part[part] = float(np.sqrt(var))
out["expected_test"] = dict(per_metric=detail, total_expected=float(exp_tot), total_naive=float(exp_tot_naive),
                            sd_total_public=sd_part["public"], sd_total_private=sd_part["private"])
print(json.dumps({k: out[k] for k in ("rows_changed", "prior_candidate_vs_s1", "expected_test")}, indent=1))
json.dump(out, open("exp/audit/verify2_postproc/v03_testfiles.json", "w"), indent=1)
