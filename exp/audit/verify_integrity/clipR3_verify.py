"""Adversarial re-derivation of the integrity claim "hard clip gas <= 1e6 and EUI <= 1e6/A".

Read-only with respect to the project: loads chain_H.npz, a few cached train/test columns and
submissions/sub_v19vce.csv, writes only exp/audit/verify_integrity/clipR3_verify.json.

Checks
 A. the chain arrays are what they claim to be (split, y on H, A, EUI = gas/A identity)
 B. the 1e6 bound is a hard truncation of the data (tail shape, bound visible on P alone)
 C. the clip effect on H (corrected chain prediction), split halves, bootstrap, rows hurt
 D. the test side: rows above the bound in sub_v19vce, evidence that test is truncated too,
    and the expected size of the gain on test (lower bound and calibrated expectation).
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, "src")
from common import holdout_split, SIGMAS  # noqa: E402

OUT = "exp/audit/verify_integrity/clipR3_verify.json"
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SG, SE = SIGMAS[GAS], SIGMAS[EUI]
C = 1e6
t0 = time.time()
res = {}


def nr(y, p, s):
    return float(np.sqrt(np.mean((y - p) ** 2)) / s)


# ---------------------------------------------------------------- A. sanity of chain arrays
d = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = d["P"], d["H"], d["A"], d["M"]
tr = pd.read_parquet("cache/train.parquet", columns=["id", GAS, EUI, "bld_dim_06", "bld_dim_12"])
n = len(tr)
P2, H2 = holdout_split(n)
g = tr[GAS].to_numpy(float)
e = tr[EUI].to_numpy(float)
res["A_sanity"] = {
    "n_train": int(n),
    "split_equals_holdout_split": bool(np.array_equal(P, P2) and np.array_equal(H, H2)),
    "P_H_disjoint_cover": bool(len(np.intersect1d(P, H)) == 0 and len(P) + len(H) == n),
    "A_vs_bld06_maxabs": float(np.nanmax(np.abs(A - tr["bld_dim_06"].to_numpy(float)))),
    "M_vs_bld12_maxabs": float(np.nanmax(np.abs(M - tr["bld_dim_12"].to_numpy(float)))),
    "gas_y_vs_train_maxabs": float(np.max(np.abs(d["gas_y"] - g[H]))),
    "eui_y_vs_train_maxabs": float(np.max(np.abs(d["eui_y"] - e[H]))),
    "eui_identity_max_rel_all_train": float(np.max(np.abs(e * A - g) / g)),
    "train_A_min": float(np.min(A)),
    "train_A_nan": int(np.isnan(A).sum()),
}

# ---------------------------------------------------------------- B. is 1e6 a hard truncation?
bins10 = np.arange(900_000, 1_000_001, 10_000)
bins50 = np.arange(500_000, 1_000_001, 50_000)
c50 = np.histogram(g, bins=bins50)[0]
c50P = np.histogram(g[P], bins=bins50)[0]
x = np.arange(len(c50))
b, a = np.polyfit(x, np.log(c50), 1)
r = float(np.exp(b))
exp_ge = float(np.exp(a + b * len(c50)) / (1 - r))
bP, aP = np.polyfit(x, np.log(c50P), 1)
rP = float(np.exp(bP))
exp_geP = float(np.exp(aP + bP * len(c50P)) / (1 - rP))
res["B_truncation"] = {
    "max_gas_all": float(g.max()),
    "max_gas_P_only": float(g[P].max()),
    "max_gas_H_only": float(g[H].max()),
    "n_gas_ge_1e6_all": int((g >= C).sum()),
    "n_gas_in_990k_1e6": int(((g >= 990_000) & (g < C)).sum()),
    "max_eui_times_A_all": float((e * A).max()),
    "train_counts_10k_bins_900k_to_1e6": np.histogram(g, bins=bins10)[0].tolist(),
    "train_counts_50k_bins_500k_to_1e6": c50.tolist(),
    "loglinear_decay_per_50k": r,
    "expected_train_rows_ge_1e6_if_tail_continued": exp_ge,
    "P_only_expected_rows_ge_1e6_if_tail_continued": exp_geP,
    "P_only_observed_rows_ge_1e6": int((g[P] >= C).sum()),
}

# ---------------------------------------------------------------- C. effect on H
AH = A[H]
NH = len(H)
rng = np.random.default_rng(20261020)
perm = rng.permutation(NH)
halves = (perm[: NH // 2], perm[NH // 2:])
NBOOT = 1000
resC = {}
for m, s, cap, label in (
    ("gas", SG, np.full(NH, C), "cap_1e6"),
    ("eui", SE, C / AH, "cap_1e6_over_A"),
    ("gas", SG, np.full(NH, float(g[P].max())), "cap_Pmax"),
    ("eui", SE, float(g[P].max()) / AH, "cap_Pmax_over_A"),
):
    y = d[f"{m}_y"]
    p = d[f"{m}_corrected"]
    pt = d[f"{m}_teacher"]
    pc = np.minimum(p, cap)
    above = p > cap
    gain = (p - y) ** 2 - (pc - y) ** 2  # >0 means clip helps that row
    base, cl = nr(y, p, s), nr(y, pc, s)
    rb = np.random.default_rng(777)  # same bootstrap draws for every configuration
    boots = []
    for _ in range(NBOOT):
        i = rb.integers(0, NH, NH)
        boots.append((nr(y[i], pc[i], s) - nr(y[i], p[i], s)) / 5)
    boots = np.array(boots)
    resC[f"{m}_{label}"] = {
        "n_above": int(above.sum()),
        "teacher_n_above": int((pt > cap).sum()),
        "base_nrmse": base,
        "clipped_nrmse": cl,
        "delta_metric": cl - base,
        "delta_total": (cl - base) / 5,
        "rows_hurt": int((gain < -1e-9).sum()),
        "sum_row_gain_sse": float(gain.sum()),
        "excess_above": sorted(np.round((p - cap)[above], 3).tolist()),
        "y_over_cap_on_above_rows": sorted(np.round((y / cap)[above], 4).tolist()),
        "half_delta_total": [(nr(y[h], pc[h], s) - nr(y[h], p[h], s)) / 5 for h in halves],
        "boot_mean": float(boots.mean()),
        "boot_sd": float(boots.std()),
        "boot_frac_ge0": float((boots >= 0).mean()),
    }
resC["claimed_pair_total"] = resC["gas_cap_1e6"]["delta_total"] + resC["eui_cap_1e6_over_A"]["delta_total"]
res["C_H_effect"] = resC

# ---------------------------------------------------------------- D. test side
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
sub = pd.read_csv("submissions/sub_v19vce.csv")
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
te = te.set_index("id")
At = te.loc[sub["id"], "bld_dim_06"].to_numpy(float)
Mt = te.loc[sub["id"], "bld_dim_12"].to_numpy(float)
pg = sub[GAS].to_numpy(float)
pe = sub[EUI].to_numpy(float)
capE = C / At
ag = pg > C
ae = pe > capE
NT = len(sub)
resD = {
    "sub_rows": int(NT),
    "sub_id_order_equals_sample": bool(np.array_equal(sub["id"].to_numpy(), ss["id"].to_numpy())),
    "test_A_min": float(At.min()),
    "test_A_nan": int(np.isnan(At).sum()),
    "n_gas_above_1e6": int(ag.sum()),
    "gas_above_values": sorted(np.round(pg[ag], 1).tolist()),
    "n_eui_above_1e6_over_A": int(ae.sum()),
    "eui_excess": sorted(np.round((pe - capE)[ae], 3).tolist()),
    "n_rows_both": int((ag & ae).sum()),
    "eui_times_A_on_gas_rows": sorted(np.round((pe * At)[ag], 0).tolist()),
}

# D1. is test truncated like train?  (a) prediction tail vs H chain prediction tail
pH = d["gas_corrected"]
tails = {}
for t in (700_000, 800_000, 900_000, 950_000, 1_000_000):
    nH = int((pH > t).sum())
    nT = int((pg > t).sum())
    tails[str(t)] = {"H_chain": nH, "test": nT, "test_expected_from_H_rate": nH * NT / NH}
resD["pred_tail_counts"] = tails

# (b) M tail: test/train ratio vs a truncated-lognormal estimate of rows removed by the cut
ratio0 = NT / n
logT = np.log(C)


def trunc_fit(xv):
    # MLE of normal(mu, sd) for xv observed only below logT
    def nll(th):
        mu, ls = th
        sd = np.exp(ls)
        return -(stats.norm.logpdf(xv, mu, sd).sum() - len(xv) * stats.norm.logcdf((logT - mu) / sd))
    th0 = np.array([xv.mean(), np.log(xv.std() + 1e-6)])
    o = optimize.minimize(nll, th0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
    mu, sd = o.x[0], np.exp(o.x[1])
    return float(mu), float(sd), float(stats.norm.sf((logT - mu) / sd))


Mtr = M
lg = np.log(g)
mbins = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 120), (120, 10_000)]
mt = []
extra_total = 0.0
for lo, hi in mbins:
    sel = (Mtr >= lo) & (Mtr < hi)
    ntr = int(sel.sum())
    nte = int(((Mt >= lo) & (Mt < hi)).sum())
    mu, sd, frac_removed = trunc_fit(lg[sel]) if ntr > 30 else (np.nan, np.nan, np.nan)
    exp_te_trunc = ntr * ratio0
    exp_te_untrunc = ntr * ratio0 / (1 - frac_removed) if ntr > 30 else np.nan
    if ntr > 30:
        extra_total += exp_te_untrunc - exp_te_trunc
    mt.append({
        "M_bin": [lo, hi], "train": ntr, "test": nte,
        "test_expected_if_test_truncated": round(exp_te_trunc, 1),
        "test_expected_if_test_untruncated": round(exp_te_untrunc, 1) if ntr > 30 else None,
        "z_vs_truncated": round((nte - exp_te_trunc) / np.sqrt(exp_te_trunc), 2) if exp_te_trunc > 0 else None,
        "est_frac_removed_by_cut": round(frac_removed, 4) if ntr > 30 else None,
        "train_frac_gas_ge_900k": round(float((g[sel] >= 900_000).mean()), 4) if ntr else None,
    })
resD["M_tail_truncation_test"] = mt
resD["M_tail_extra_test_rows_expected_if_untruncated"] = extra_total
selM = Mtr >= 80
resD["M_ge_80"] = {"train": int(selM.sum()), "test": int((Mt >= 80).sum()),
                   "test_expected_if_truncated": float(selM.sum() * ratio0)}

# D2. expected gain on test (full test; public/private are random subsets, same expectation)
mse0_g = (resC["gas_cap_1e6"]["base_nrmse"] * SG) ** 2  # test RMSE ~ H RMSE level
mse0_e = (resC["eui_cap_1e6_over_A"]["base_nrmse"] * SE) ** 2
# calibration E[y|p] = k*p for high predictions, estimated on H chain (not on test)
yH, eH = d["gas_y"], d["eui_y"]
peH = d["eui_corrected"]
cal = {}
for lab, msk in (("p_gt_1e6", pH > C), ("p_gt_950k", pH > 950_000), ("p_gt_900k", pH > 900_000)):
    cal[f"gas_{lab}"] = {"n": int(msk.sum()), "k_sum_ratio": float(yH[msk].sum() / pH[msk].sum())}
mskE = peH * AH > C
cal["eui_p_gt_cap"] = {"n": int(mskE.sum()), "k_sum_ratio": float(eH[mskE].sum() / peH[mskE].sum())}
mskE2 = peH * AH > 950_000
cal["eui_p_gt_0.95cap"] = {"n": int(mskE2.sum()), "k_sum_ratio": float(eH[mskE2].sum() / peH[mskE2].sum())}
# P-side calibration (independent of H labels) from P fitted prediction / residual
ypP = d["gas_pP"] + d["gas_rP"]
resD["P_side_y_reconstruct_maxabs"] = float(np.max(np.abs(ypP - g[P])))
mP = d["gas_pP"] > C
cal["gas_P_side_p_gt_1e6"] = {"n": int(mP.sum()), "k_sum_ratio": float(ypP[mP].sum() / d["gas_pP"][mP].sum())}
yeP = d["eui_pP"] + d["eui_rP"]
resD["P_side_eui_reconstruct_maxabs"] = float(np.max(np.abs(yeP - e[P])))
mPe = d["eui_pP"] * A[P] > C
cal["eui_P_side_p_gt_cap"] = {"n": int(mPe.sum()), "k_sum_ratio": float(yeP[mPe].sum() / d["eui_pP"][mPe].sum())}
resD["calibration"] = cal


def test_delta(p, cap, mse0, s, k):
    above = p > cap
    pa, ca = p[above], cap[above] if np.ndim(cap) else np.full(above.sum(), cap)
    if k is None:  # worst case for the gain: y exactly at the bound
        dsse = float(np.sum((pa - ca) ** 2))
    else:
        ey = np.minimum(k * pa, ca)
        dsse = float(np.sum((pa - ca) * (pa + ca - 2 * ey)))
    d_rmse = np.sqrt(mse0 - dsse / NT) - np.sqrt(mse0)
    return dsse, float(d_rmse / s / 5)


est = {}
for lab, kg, ke in (
    ("lower_bound_y_at_cap", None, None),
    ("H_cal_p_gt_1e6", cal["gas_p_gt_1e6"]["k_sum_ratio"], cal["eui_p_gt_cap"]["k_sum_ratio"]),
    ("H_cal_p_gt_950k", cal["gas_p_gt_950k"]["k_sum_ratio"], cal["eui_p_gt_0.95cap"]["k_sum_ratio"]),
    ("P_side_cal_p_gt_1e6", cal["gas_P_side_p_gt_1e6"]["k_sum_ratio"], cal["eui_P_side_p_gt_cap"]["k_sum_ratio"]),
):
    dg = test_delta(pg, C, mse0_g, SG, kg)
    de = test_delta(pe, capE, mse0_e, SE, ke)
    est[lab] = {"gas_dsse": dg[0], "gas_delta_total": dg[1], "eui_dsse": de[0], "eui_delta_total": de[1],
                "total": dg[1] + de[1]}
resD["test_expected_delta_total"] = est
# probability that none of the 7 gas rows falls in the public 30%
resD["prob_no_gas_row_in_public"] = float(0.7 ** int(ag.sum()))
res["D_test"] = resD
res["seconds"] = round(time.time() - t0, 1)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({
    "split_ok": res["A_sanity"]["split_equals_holdout_split"],
    "bound": res["B_truncation"]["max_gas_P_only"],
    "exp_ge_1e6": round(exp_ge, 1),
    "gas_dT": resC["gas_cap_1e6"]["delta_total"],
    "eui_dT": resC["eui_cap_1e6_over_A"]["delta_total"],
    "pair": resC["claimed_pair_total"],
    "test_n": [resD["n_gas_above_1e6"], resD["n_eui_above_1e6_over_A"]],
    "test_est": {k: round(v["total"], 9) for k, v in est.items()},
    "sec": res["seconds"],
}, indent=1))
