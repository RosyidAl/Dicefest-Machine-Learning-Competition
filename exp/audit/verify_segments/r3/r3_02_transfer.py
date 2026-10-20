"""verify_segments r3: does the H gain of the cap calibration transfer to the test rows?

1. Candidate file check (value level): candidate = cal(sub_v19vce, k=0.08) on gas and on eui via eui*A, dup rows and the
   other columns untouched.
2. Prediction-level distribution near the cap: test (sub_v19vce, non-dup rows) vs H corrected / H teacher / P teacher.
3. Expected test gain, three ways (all as TOTAL-score change; base MSE taken from H corrected):
   (m) model-implied: if y | p follows the truncated normal of the calibration with k_true, E[dSE] = d^2 - 2 d b,
       d = cal(p, 0.08) - p, b = cal(p, k_true) - p; summed over test rows for k_true in a range;
   (e) empirical bias curve: b(p) = binned mean residual of H corrected (honest), E[dSE] = d^2 - 2 d b on test rows;
       bootstrap over H rows for its sd;
   (w) test-like bootstrap: H rows resampled with importance weights (test / H density per prediction bin) at
       public (55,196) and private (128,791) size.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/r3/r3_02_transfer.py \
         > exp/audit/verify_segments/r3/r3_02_transfer.log 2>&1
"""
import hashlib
import json
import resource
import time

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

T0 = time.time()
SIG = {"gas": 151750.04, "eui": 40.65}
CAP, THR, K = 1e6, 6e5, 0.08
L2PI = 0.5 * np.log(2 * np.pi)
GAS, EL, EUI, CL, AD = ("target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm",
                        "target_clean_building_ratio", "target_electric_heating_adoption")
TC = [GAS, EL, EUI, CL, AD]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]


def cal(g, k):
    g = np.asarray(g, dtype=float)
    out = g.copy()
    m = g > THR
    s = k * g[m]
    a = (CAP - g[m]) / s
    out[m] = g[m] - s * np.exp(-0.5 * a * a - L2PI - log_ndtr(a))
    return out


def cal_eui(e, Ae, k):
    g = e * Ae
    gn = cal(g, k)
    return np.where(g > THR, e * gn / np.where(g > THR, g, 1.0), e)


# ------------------------------------------------------------------ 1. candidate check
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TC)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
ref = tr.groupby(KEY, sort=False)[TC].first().reset_index()
dup = te[KEY].merge(ref, on=KEY, how="left")[TC].values
isdup = ~np.isnan(dup[:, 0])
At = te["bld_dim_06"].values.astype(float)
del tr, ref
base = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
cand_path = "exp/audit/segments/sub_v19vce_cap_k0.08.csv"
cand = pd.read_csv(cand_path, float_precision="round_trip")
sha = hashlib.sha256(open(cand_path, "rb").read()).hexdigest()
assert (base["id"].values == te["id"].values).all() and (cand["id"].values == te["id"].values).all()
g0, e0 = base[GAS].values, base[EUI].values
g_my = np.where(isdup, g0, cal(g0, K))
e_my = np.where(isdup, e0, cal_eui(e0, At, K))
print(f"dup rows {isdup.sum()}; dup rows equal to train values in base: "
      f"{bool(np.allclose(base.loc[isdup, TC].values, dup[isdup], rtol=0, atol=0))}")
print(f"candidate sha256 {sha}")
print(f"gas: max |cand - my cal| {np.max(np.abs(cand[GAS].values - g_my)):.3e}; eui max rel "
      f"{np.max(np.abs(cand[EUI].values - e_my) / e_my):.3e}; other columns identical: "
      f"{all(np.array_equal(cand[c].values, base[c].values) for c in (EL, CL, AD))}; dup rows untouched: "
      f"{bool(np.array_equal(cand.loc[isdup, TC].values, base.loc[isdup, TC].values))}")
dg = cand[GAS].values - g0
print(f"gas rows changed {int((dg != 0).sum())}, by > 0.1 sigma {int((np.abs(dg) > 0.1 * SIG['gas']).sum())}, "
      f"max |change| {np.abs(dg).max():.0f}, max before {g0.max():.0f} after {cand[GAS].values.max():.0f}; "
      f"rows > 1e6 before {int((g0 > CAP).sum())}")

# ------------------------------------------------------------------ 2. distributions
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH, AP = A[H], A[P]
yH, cH, tH = z["gas_y"], z["gas_corrected"], z["gas_teacher"]
yHe, cHe = z["eui_y"], z["eui_corrected"]
pP, yP = z["gas_pP"], z["gas_pP"] + z["gas_rP"]
live = ~isdup
gT, eT, AT = g0[live], e0[live], At[live]
nT = len(g0)  # metric runs over all test rows; dup rows have zero change
EDG = np.array([0.6, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 9.0]) * 1e6
lab = [f"[{EDG[i] / 1e6:.2f},{EDG[i + 1] / 1e6:.2f})" for i in range(len(EDG) - 1)]


def cnt(p):
    return np.histogram(p, EDG)[0]


views = {"test gas": (gT, len(g0)), "H corrected": (cH, len(H)), "H teacher": (tH, len(H)),
         "P teacher": (pP, len(P)), "test eui*A": (eT * AT, len(g0)), "H corr eui*A": (cHe * AH, len(H))}
print("\nrows per 100k by bin of prediction / 1e6 (counts in brackets)")
for nm, (p, n) in views.items():
    c = cnt(p)
    print(f"  {nm:13s} " + " ".join(f"{lab[i]} {1e5 * c[i] / n:6.1f} ({c[i]:4d})" for i in range(len(c))))
for thr in (0.9e6, 0.95e6, 1e6):
    nt, nh = int((gT > thr).sum()), int((cH > thr).sum())
    exp_t = nh / len(H) * len(g0)
    print(f"  gas pred > {thr / 1e6:.2f}e6: test {nt} vs expected at the H rate {exp_t:.1f}  "
          f"(Poisson z {(nt - exp_t) / np.sqrt(exp_t):+.2f})")


# ------------------------------------------------------------------ 3. expected test gain
def to_total(dsse, n, nr0, m):
    mse0 = (nr0 * SIG[m]) ** 2
    return (np.sqrt(mse0 + dsse / n) - np.sqrt(mse0)) / SIG[m] / 5


nr0g = float(np.sqrt(np.mean((yH - cH) ** 2)) / SIG["gas"])
nr0e = float(np.sqrt(np.mean((yHe - cHe) ** 2)) / SIG["eui"])
out = {}
# realised on H with k = 0.08 (reference)
dH = cal(cH, K) - cH
dHe = cal_eui(cHe, AH, K) - cHe
real_g = to_total(np.sum((yH - cH - dH) ** 2 - (yH - cH) ** 2), len(H), nr0g, "gas")
real_e = to_total(np.sum((yHe - cHe - dHe) ** 2 - (yHe - cHe) ** 2), len(H), nr0e, "eui")
print(f"\nrealised on H (k 0.08): gas {real_g:+.8f} eui {real_e:+.8f} total {real_g + real_e:+.8f}")
# (m) model-implied
print("(m) model-implied expected change (truncated-normal truth with k_true; applied k 0.08):")
dT = cal(gT, K) - gT
dTe = cal_eui(eT, AT, K) - eT
for kt in (0.06, 0.07, 0.075, 0.08, 0.085, 0.09):
    bH = cal(cH, kt) - cH
    bHe = cal_eui(cHe, AH, kt) - cHe
    bT = cal(gT, kt) - gT
    bTe = cal_eui(eT, AT, kt) - eT
    mH = to_total(np.sum(dH ** 2 - 2 * dH * bH), len(H), nr0g, "gas")
    mHe = to_total(np.sum(dHe ** 2 - 2 * dHe * bHe), len(H), nr0e, "eui")
    mT = to_total(np.sum(dT ** 2 - 2 * dT * bT), nT, nr0g, "gas")
    mTe = to_total(np.sum(dTe ** 2 - 2 * dTe * bTe), nT, nr0e, "eui")
    out[f"m_k{kt}"] = dict(H_gas=mH, H_eui=mHe, test_gas=mT, test_eui=mTe)
    print(f"   k_true {kt:.3f}: H gas {mH:+.8f} eui {mHe:+.8f} | test gas {mT:+.8f} eui {mTe:+.8f} "
          f"total {mT + mTe:+.8f}")

# (e) empirical bias curve from H corrected (honest residuals), bootstrap over H rows
EB = np.array([0.6, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 9.0]) * 1e6


def bias_bins(y, p, idx=None):
    if idx is not None:
        y, p = y[idx], p[idx]
    b = np.digitize(p, EB) - 1
    res = np.zeros(len(EB) - 1)
    for j in range(len(EB) - 1):
        m = b == j
        res[j] = (y[m] - p[m]).mean() if m.any() else 0.0
    return res


def exp_gain(bvec, p, d):
    b = np.digitize(p, EB) - 1
    ok = (b >= 0) & (b < len(EB) - 1)
    bb = np.zeros(len(p))
    bb[ok] = bvec[b[ok]]
    return np.sum(d ** 2 - 2 * d * bb)


gTe_A = eT * AT
bvec_g = bias_bins(yH, cH)
bvec_e = bias_bins(yHe * AH, cHe * AH)  # in gas units
# eui change in gas units: d_gas = dTe * AT; convert dSSE back to eui units by / AT^2 per row
dTe_gas = dTe * AT


def exp_gain_eui(bvec):
    b = np.digitize(gTe_A, EB) - 1
    ok = (b >= 0) & (b < len(EB) - 1)
    bb = np.zeros(len(gTe_A))
    bb[ok] = bvec[b[ok]]
    return np.sum((dTe_gas ** 2 - 2 * dTe_gas * bb) / AT ** 2)


eg = to_total(exp_gain(bvec_g, gT, dT), nT, nr0g, "gas")
ee = to_total(exp_gain_eui(bvec_e), nT, nr0e, "eui")
rng = np.random.default_rng(7)
bsg, bse = [], []
for _ in range(1000):
    i = rng.integers(0, len(H), len(H))
    bsg.append(to_total(exp_gain(bias_bins(yH, cH, i), gT, dT), nT, nr0g, "gas"))
    bse.append(to_total(exp_gain_eui(bias_bins(yHe * AH, cHe * AH, i)), nT, nr0e, "eui"))
bsg, bse = np.array(bsg), np.array(bse)
print(f"(e) empirical bias curve of H corrected per bin (gas units): {np.round(bvec_g).astype(int).tolist()}")
print(f"    expected test change: gas {eg:+.8f} (boot mean {bsg.mean():+.8f} sd {bsg.std():.8f}) eui {ee:+.8f} "
      f"(boot sd {bse.std():.8f}); total {eg + ee:+.8f}  P(total<0) {np.mean(bsg + bse < 0):.3f}")
# same with the P-teacher bias curve (independent rows; single-model OOF teacher)
bvec_gP = bias_bins(yP, pP)
egP = to_total(exp_gain(bvec_gP, gT, dT), nT, nr0g, "gas")
print(f"    with the P-teacher bias curve {np.round(bvec_gP).astype(int).tolist()}: test gas {egP:+.8f}")
out["e"] = dict(test_gas=eg, test_eui=ee, boot_sd_gas=bsg.std(), boot_sd_eui=bse.std(), test_gas_Pcurve=egP)

# (w) test-like importance-weighted bootstrap of H rows
def wts(pv, pt):
    ch, ct = np.histogram(pv, EB)[0], np.histogram(pt, EB)[0]
    r = np.where(ch > 0, (ct / nT) / np.maximum(ch / len(pv), 1e-12), 0.0)
    b = np.digitize(pv, EB) - 1
    w = np.ones(len(pv))
    ok = (b >= 0) & (b < len(EB) - 1)
    w[ok] = r[b[ok]]
    # rows below 0.6e6 keep weight that makes the total mass match
    below = ~ok
    w[below] = (1 - ct.sum() / nT) / max(below.mean(), 1e-12)
    return w / w.sum()


wg = wts(cH, gT)
e0g, e1g = (yH - cH) ** 2, (yH - cH - dH) ** 2
e0e, e1e = (yHe - cHe) ** 2, (yHe - cHe - dHe) ** 2
# weighted own-rows value
wv_g = (np.sqrt(np.sum(wg * e1g)) - np.sqrt(np.sum(wg * e0g))) / SIG["gas"] / 5
wv_e = (np.sqrt(np.sum(wg * e1e)) - np.sqrt(np.sum(wg * e0e))) / SIG["eui"] / 5
print(f"(w) importance-weighted H value (test prediction mix): gas {wv_g:+.8f} eui {wv_e:+.8f} total {wv_g + wv_e:+.8f}")
cw = np.cumsum(wg)
for nm, sz in (("public", 55196), ("private", 128791)):
    v = []
    for _ in range(3000):
        i = np.minimum(np.searchsorted(cw, rng.random(sz)), len(cw) - 1)
        v.append(((np.sqrt(e1g[i].mean()) - np.sqrt(e0g[i].mean())) / SIG["gas"]
                  + (np.sqrt(e1e[i].mean()) - np.sqrt(e0e[i].mean())) / SIG["eui"]) / 5)
    v = np.array(v)
    out[f"w_{nm}"] = [float(v.mean()), float(v.std()), float(np.mean(v < 0))]
    print(f"    {nm:7s} size {sz}: mean {v.mean():+.8f} sd {v.std():.8f} P(<0) {np.mean(v < 0):.3f} "
          f"P(<= -1e-5) {np.mean(v <= -1e-5):.3f}  5% {np.quantile(v, .05):+.8f} 95% {np.quantile(v, .95):+.8f}")
json.dump(out, open("exp/audit/verify_segments/r3/r3_02_transfer.json", "w"), indent=1, default=float)
print(f"elapsed {time.time() - T0:.1f} s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
