"""verify_postproc/adv v02: was H lucky, and does the gain transfer to the test rows?

For a shift Delta_i of row i with residual r_i = y_i - p_i:  dSE_i = Delta_i^2 - 2 Delta_i r_i.
  S1 (Delta = d p, group bias model E[r] = b p):          E[dSE_i] = p_i^2 d (d - 2 b)   (= -d^2 p^2 when b = d)
  S3 (Delta = k x, calibration model E[r] = k* x):         E[dSE_i] = k x_i (k x_i - 2 k* x_i)  (= -k^2 x^2 when k* = k)
  noise of the realised sum:  Var(sum dSE) ~ sum 4 Delta_i^2 Var(r_i), Var(r_i) ~ p_i^2 (a0 + b0 / M_i)
Compares (1) realised vs model-expected gains on H, and (2) model-expected gains on the H rows vs the actual test rows
(sub_v19vce predictions, full-train OOF deltas, k from all of H, bounds from all train rows) -> transfer ratio.
NRMSE change from MSE change: dNRMSE = dMSE / (2 RMSE) / sigma^2 with the H RMSE of the base.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/adv/v02_expected_vs_realized.py
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import norm

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[:3]
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/adv"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
y = {m: z[f"{m}_y"] for m in MET}
pP = {m: z[f"{m}_pP"] for m in MET}
rP = {m: z[f"{m}_rP"] for m in MET}
base = {m: np.maximum(z[f"{m}_corrected"], 0.0) for m in MET}
GROUPS = ((2, 3), (4, 4))
gP = [(MP >= a) & (MP <= b) for a, b in GROUPS]
gH = [(MH >= a) & (MH <= b) for a, b in GROUPS]
dP = {m: [float(pP[m][g] @ rP[m][g] / (pP[m][g] @ pP[m][g])) for g in gP] for m in MET}
fo = json.load(open("exp/audit/postproc/12_fullvc_smallM.json"))
dF = {m: [fo[m]["M<=3"]["ls_delta"], fo[m]["M=4"]["ls_delta"]] for m in MET}
VAR = {}
for m in MET:
    X = np.column_stack([np.ones(len(P)), 1.0 / MP])
    VAR[m] = np.linalg.lstsq(X, (rP[m] / pP[m]) ** 2, rcond=None)[0]
rmse0 = {m: float(np.sqrt(np.mean((y[m] - base[m]) ** 2))) for m in MET}


def to_nrmse(sum_dse, n, m):
    return sum_dse / n / (2 * rmse0[m]) / SIGMAS[COL[m]]


def trunc_shift(p, s, U):
    b = (U - p) / s
    return -s * np.exp(norm.logpdf(b) - log_ndtr(b))


res = {"dP": dP, "dF": dF}
print("deltas P-OOF", {m: np.round(v, 4).tolist() for m, v in dP.items()}, "| full-train OOF",
      {m: np.round(v, 4).tolist() for m, v in dF.items()})

# ---------------------------------------------------------------- (1) H: realised vs expected S1 gain
nH = len(H)
print("\n(1) S1 on H (P-OOF deltas applied to the corrected H predictions)")
tot_real = tot_exp = tot_var = 0.0
for m in MET:
    for j, (g, d) in enumerate(zip(gH, dP[m])):
        p, r = base[m][g], (y[m] - base[m])[g]
        D = d * p
        real = float(np.sum(D ** 2 - 2 * D * r))
        exp_ = float(np.sum(p ** 2 * d * (d - 2 * d)))           # b = d (P-OOF bias)
        exp_f = float(np.sum(p ** 2 * d * (d - 2 * dF[m][j])))    # b = full-train OOF bias
        sd = float(np.sqrt(np.sum(4 * D ** 2 * p ** 2 * (VAR[m][0] + VAR[m][1] / MH[g]))))
        # realised mean relative residual of H rows in the group (what H 'saw')
        bH = float(p @ r / (p @ p))
        cv = lambda v: to_nrmse(v, nH, m) / 5  # noqa: E731
        print(f"   {m:4s} {GROUPS[j]} n {g.sum():4d} H-realised bias {bH:+.3f} | TOTAL realised {cv(real):+.8f} "
              f"expected(b=dP) {cv(exp_):+.8f} expected(b=dF) {cv(exp_f):+.8f} noise sd {cv(sd):.8f}")
        res[f"H_S1_{m}_{j}"] = dict(n=int(g.sum()), bias_H=bH, real=cv(real), exp_bP=cv(exp_), exp_bF=cv(exp_f), sd=cv(sd))
        tot_real += cv(real)
        tot_exp += cv(exp_f)
        tot_var += cv(sd) ** 2
print(f"   S1 TOTAL realised {tot_real:+.8f} expected(b=full OOF) {tot_exp:+.8f} noise sd {np.sqrt(tot_var):.8f} "
      f"-> z of (realised - expected) {(tot_real - tot_exp) / np.sqrt(tot_var):+.2f}")
res["H_S1_total"] = dict(real=tot_real, exp=tot_exp, sd=float(np.sqrt(tot_var)))

# ---------------------------------------------------------------- (1b) H: realised vs expected S3 gain (after S1)
print("\n(1b) S3 on H after S1 (k fitted on all of H, i.e. the expectation uses k* = k)")
gmaxP = (pP["gas"] + rP["gas"]).max()
kH, xH = {}, {}
for m in ("gas", "eui"):
    p = base[m].copy()
    for g, d in zip(gH, dP[m]):
        p[g] *= 1 + d
    s = p * np.sqrt(np.maximum(VAR[m][0] + VAR[m][1] / MH, 1e-8))
    x = trunc_shift(p, s, gmaxP if m == "gas" else gmaxP / AH)
    k = float(x @ (y[m] - p) / (x @ x))
    kH[m], xH[m] = k, x
    real = float(np.sum((k * x) ** 2 - 2 * k * x * (y[m] - p)))
    exp_ = float(-np.sum((k * x) ** 2))
    sd = float(np.sqrt(np.sum(4 * (k * x) ** 2 * p ** 2 * (VAR[m][0] + VAR[m][1] / MH))))
    print(f"   {m:4s} k {k:.4f} rows |kx|>1e-3 sigma {int(np.sum(np.abs(k * x) > 1e-3 * SIGMAS[COL[m]]))} | TOTAL "
          f"realised(in-sample k) {to_nrmse(real, nH, m) / 5:+.8f} expected {to_nrmse(exp_, nH, m) / 5:+.8f} "
          f"noise sd {to_nrmse(sd, nH, m) / 5:.8f}")
    res[f"H_S3_{m}"] = dict(k=k, real=to_nrmse(real, nH, m) / 5, exp=to_nrmse(exp_, nH, m) / 5, sd=to_nrmse(sd, nH, m) / 5)

# ---------------------------------------------------------------- (2) test rows: expected gains under the same models
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
ref = tr.groupby(KEY, sort=False)[[GAS]].first().reset_index()
live = np.isnan(te[KEY].merge(ref, on=KEY, how="left")[GAS].values)
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
gmaxF = tr[GAS].max()
nT = len(te)
print(f"\n(2) test: rows {nT}, duplicates (not changed) {int((~live).sum())}, M groups "
      f"{[(int(((Mt >= a) & (Mt <= b) & live).sum())) for a, b in GROUPS]} (H {[int(g.sum()) for g in gH]})")
tot_t = tot_h = 0.0
for m in MET:
    pt = sub[COL[m]].values.astype(float)
    for j, (a, b) in enumerate(GROUPS):
        g = (Mt >= a) & (Mt <= b) & live
        e_t = to_nrmse(float(np.sum(-(dF[m][j] * pt[g]) ** 2)), nT, m) / 5
        e_h = to_nrmse(float(np.sum(-(dP[m][j] * base[m][gH[j]]) ** 2)), nH, m) / 5
        q = lambda v: np.round(np.percentile(v, [25, 50, 75]), 1).tolist()  # noqa: E731
        print(f"   S1 {m:4s} {GROUPS[j]}: expected TOTAL test {e_t:+.8f} vs H {e_h:+.8f} (ratio {e_t / e_h:.2f}) | "
              f"rms p test {np.sqrt(np.mean(pt[g] ** 2)):.4g} H {np.sqrt(np.mean(base[m][gH[j]] ** 2)):.4g} | "
              f"n/N test {g.sum() / nT:.5f} H {gH[j].sum() / nH:.5f} | p quartiles test {q(pt[g])} H {q(base[m][gH[j]])}")
        res[f"T_S1_{m}_{j}"] = dict(exp_test=e_t, exp_H=e_h)
        tot_t += e_t
        tot_h += e_h
    if m in kH:
        p = pt.copy()
        for j, (a, b) in enumerate(GROUPS):
            g = (Mt >= a) & (Mt <= b) & live
            p[g] *= 1 + dF[m][j]
        s = p * np.sqrt(np.maximum(VAR[m][0] + VAR[m][1] / Mt, 1e-8))
        x = np.where(live, trunc_shift(p, s, gmaxF if m == "gas" else gmaxF / At), 0.0)
        e_t = to_nrmse(float(-np.sum((kH[m] * x) ** 2)), nT, m) / 5
        e_h = res[f"H_S3_{m}"]["exp"]
        thr = 1e-3 * SIGMAS[COL[m]]
        print(f"   S3 {m:4s}: expected TOTAL test {e_t:+.8f} vs H {e_h:+.8f} (ratio {e_t / e_h:.2f}) | rows |kx|>1e-3 "
              f"sigma per 1e5: test {1e5 * np.mean(np.abs(kH[m] * x) > thr):.1f} H "
              f"{1e5 * np.mean(np.abs(kH[m] * xH[m]) > thr):.1f} | sum (kx)^2/N/sigma^2 test "
              f"{np.sum((kH[m] * x) ** 2) / nT / SIGMAS[COL[m]] ** 2:.3e} H "
              f"{np.sum((kH[m] * xH[m]) ** 2) / nH / SIGMAS[COL[m]] ** 2:.3e}")
        res[f"T_S3_{m}"] = dict(exp_test=e_t, exp_H=e_h)
        tot_t += e_t
        tot_h += e_h
        # prediction mass near the cap: share of rows with gas-scale prediction above 0.9 / 0.95 of the cap
        if m == "gas":
            for f in (0.8, 0.9, 0.95, 1.0):
                print(f"      share of rows with base gas pred > {f:.2f} cap: test {np.mean(pt[live] > f * gmaxF):.6f} "
                      f"H {np.mean(base['gas'] > f * gmaxP):.6f}; truth share in train > {f:.2f} cap "
                      f"{np.mean(tr[GAS].values > f * gmaxF):.6f}")
print(f"   expected S1+S3 TOTAL: test {tot_t:+.8f} vs H {tot_h:+.8f}")
res["expected_S1S3_test"] = tot_t
res["expected_S1S3_H"] = tot_h
json.dump(res, open(f"{OUT}/v02_expected_vs_realized.json", "w"), indent=1, default=float)
