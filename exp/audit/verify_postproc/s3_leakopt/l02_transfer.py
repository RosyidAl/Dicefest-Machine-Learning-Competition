"""Adversarial check (leakage / optimism lens) of the S3 soft cap: does the H gain transfer to the test rows?

A. Is the test split truncated at gas < 1e6 as well? Label-free: a low-capacity OLS model of log gas on building
   features (fitted on train; ~30 coefficients on 429k rows, so in-sample optimism is negligible) scores train and
   test rows; under a common truncation the test/train count ratio is the same (n_test/n_train) in every bin of the
   predicted gas, while a train-only truncation leaves an excess of test rows in the top bins. Expected excess under
   the train-only alternative from a log-normal tail model.
B. Test-side profile of the rule on the actual sub_v19vce predictions (after S1 with the full-OOF deltas, excluding
   the 77 duplicate rows): standardized distance to the cap, size of the change, versus H.
C. Projected test change. Under E[r | x] = k* x: E[dMSE] = (k^2 - 2 k k*) mean(x^2). k* for the test regime
   (10-fold-averaged models trained on all train rows) is scaled from H by the ratio k*(full-train OOF) / k*(P-OOF),
   i.e. the effect of 386k vs 270k training rows measured in the same single-model OOF regime. Also a
   non-parametric bin-wise projection (H per-row gain by distance bin x test bin counts).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/s3_leakopt/l02_transfer.py
"""
import json
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

t0 = time.time()
OUT = "exp/audit/verify_postproc/s3_leakopt"
GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
COL = {"gas": GAS, "eui": EUI}
MET = ("gas", "eui")
KCL = {"gas": 0.5046, "eui": 0.6574}
res = {}
mem_kb = int([ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0])
assert mem_kb >= 9_000_000, f"MemAvailable {mem_kb} kB < 9e6, abort"

KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
ERA = [f"bld_era_{i:02d}" for i in range(1, 16)]
NUM = ["bld_dim_01", "bld_dim_02", "bld_dim_03", "bld_dim_06", "bld_dim_12", "bld_dim_17", "env_climate_01"]
cols = sorted(set(KEY + ERA + NUM))
tr = pd.read_parquet("cache/train.parquet", columns=cols + [GAS, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id"] + cols)
nTr, nTe = len(tr), len(te)
RATIO = nTe / nTr
GMAX = float(tr[GAS].max())
print(f"train {nTr} test {nTe} ratio {RATIO:.5f}; train gas max {GMAX:.4f}; rows >= 1e6: {int((tr[GAS] >= 1e6).sum())}")


# ---------------- A. truncation in test?
def design(df):
    X = [np.ones(len(df))]
    for c in ("bld_dim_01", "bld_dim_03", "bld_dim_06", "bld_dim_12"):
        X.append(np.log(np.clip(df[c].values.astype(float), 1e-3, None)))
    for c in ("bld_dim_02", "bld_dim_17", "env_climate_01"):
        X.append(df[c].values.astype(float))
    X.append(np.log(df["bld_dim_06"].values / df["bld_dim_12"].values))
    for c in ERA:
        X.append(df[c].values.astype(float))
    return np.column_stack(X)


Xtr, Xte = design(tr), design(te)
mu, sd = np.nanmean(Xtr[:, 1:], 0), np.nanstd(Xtr[:, 1:], 0) + 1e-12
Xtr[:, 1:], Xte[:, 1:] = (Xtr[:, 1:] - mu) / sd, (Xte[:, 1:] - mu) / sd
Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
ylog = np.log(tr[GAS].values)
beta = np.linalg.lstsq(Xtr, ylog, rcond=None)[0]
ftr, fte = Xtr @ beta, Xte @ beta
rsd = float(np.std(ylog - ftr))
print(f"OLS log-gas proxy: {Xtr.shape[1]} coefs, R2 {1 - np.var(ylog - ftr) / np.var(ylog):.4f}, resid sd {rsd:.4f}")
# split-sample check of in-sample optimism of the proxy (fit on half the train rows, compare residual sd)
hh = np.random.default_rng(0).permutation(nTr)[: nTr // 2]
mask = np.zeros(nTr, bool)
mask[hh] = True
b2 = np.linalg.lstsq(Xtr[mask], ylog[mask], rcond=None)[0]
print(f"   half-fit resid sd in-sample {np.std(ylog[mask] - Xtr[mask] @ b2):.5f} vs other half {np.std(ylog[~mask] - Xtr[~mask] @ b2):.5f}")
zt, ze = (np.log(1e6) - ftr) / rsd, (np.log(1e6) - fte) / rsd  # standardized distance of the proxy to the cap
print("bin of proxy distance to log(1e6) in resid-sd | n train | n test | test/train / expected | z | "
      "expected test/train ratio if ONLY train were truncated")
rowsA = []
for lo, hi in ((-9, 0), (0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 99)):
    st, se_ = (zt >= lo) & (zt < hi), (ze >= lo) & (ze < hi)
    nt, ne = int(st.sum()), int(se_.sum())
    exp_ = nt * RATIO
    zsc = (ne - exp_) / np.sqrt(exp_ * (1 - nTe / (nTr + nTe))) if nt else np.nan
    # alternative: test untruncated; fraction of the untruncated population above 1e6 in this bin is q, train keeps 1-q
    # q estimated from the proxy's log-normal tail: mean over the bin's test rows of P(log gas > log 1e6)
    q = float(np.mean(norm.sf(ze[se_]))) if ne else np.nan
    alt = RATIO / (1 - q) if ne else np.nan
    rowsA.append(dict(bin=[lo, hi], n_train=nt, n_test=ne, rel=ne / exp_ if nt else None, z=zsc, alt_rel=alt / RATIO))
    print(f"   [{lo:4},{hi:3}) {nt:7d} {ne:7d}  {ne / max(exp_, 1e-9):6.3f}  {zsc:+5.2f}  alt {alt / RATIO:6.3f}")
for c, edges in (("bld_dim_06", (5000, 6000, 7000, 8000, 10000, 1e9)), ("bld_dim_12", (60, 80, 100, 120, 1e9))):
    v1, v2 = tr[c].values, te[c].values
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        nt, ne = int(((v1 >= lo) & (v1 < hi)).sum()), int(((v2 >= lo) & (v2 < hi)).sum())
        out.append(f"[{lo:g},{hi:g}) {nt}/{ne} rel {ne / (nt * RATIO):.3f}")
    print(f"   {c}: " + "; ".join(out))
# pooled top-tail test: rows with proxy distance < 1.5
st, se_ = zt < 1.5, ze < 1.5
nt, ne = int(st.sum()), int(se_.sum())
q = float(np.mean(norm.sf(ze[se_])))
print(f"pooled proxy dist < 1.5: train {nt} test {ne} expected {nt * RATIO:.1f} (rel {ne / (nt * RATIO):.3f}); "
      f"train-only-truncation alternative expects rel {1 / (1 - q):.3f} (q {q:.3f}), i.e. {nt * RATIO / (1 - q):.1f} test rows")
res["A_truncation"] = dict(bins=rowsA, pooled=dict(n_train=nt, n_test=ne, rel=ne / (nt * RATIO), alt_rel=1 / (1 - q)))

# ---------------- B. test-side profile of the rule
ref = tr.groupby(KEY, sort=False)[[GAS, EUI]].first().reset_index()
dup = te[KEY].merge(ref, on=KEY, how="left")[[GAS, EUI]].values
isdup = ~np.isnan(dup[:, 0])
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
print("duplicate rows", int(isdup.sum()), "| dup rows already equal to train targets:",
      bool(np.allclose(sub.loc[isdup, [GAS, EUI]].values, dup[isdup], rtol=0, atol=0)))
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
fo = json.load(open("exp/audit/postproc/12_fullvc_smallM.json"))
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
SD = {}
for m in MET:
    pp, rr = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pp)), 1.0 / MP]), (rr / pp) ** 2, rcond=None)[0]
GMAX_P = float((z["gas_pP"] + z["gas_rP"]).max())


def mills_shift(p, s, T):
    b = (T - p) / s
    return -s * np.exp(norm.logpdf(b) - norm.logcdf(b))


def sdev(m, p, Mv):
    return p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / Mv, 1e-8, None))


# H after S1 (P-OOF deltas), as in the validated chain
pH, xHd, bH, rH = {}, {}, {}, {}
for m in MET:
    p = np.clip(z[f"{m}_corrected"], 0, None)
    pp, rr = z[f"{m}_pP"], z[f"{m}_rP"]
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + float(pp[gP] @ rr[gP] / (pp[gP] @ pp[gP]))
    s = sdev(m, p, MH)
    T = GMAX_P if m == "gas" else GMAX_P / AH
    pH[m], xHd[m], bH[m], rH[m] = p, mills_shift(p, s, T), (T - p) / s, z[f"{m}_y"] - p
# test after S1 (full-OOF deltas), non-duplicate rows
pT, xT, bT = {}, {}, {}
live = ~isdup
for m in MET:
    p = sub[COL[m]].values.astype(float).copy()
    for key, g in (("M<=3", (Mt >= 2) & (Mt <= 3)), ("M=4", Mt == 4)):
        p[g & live] *= 1 + fo[m][key]["ls_delta"]
    s = sdev(m, p, Mt)
    T = GMAX if m == "gas" else GMAX / At
    pT[m], xT[m], bT[m] = p[live], mills_shift(p, s, T)[live], ((T - p) / s)[live]
nH, nT = len(H), int(live.sum())
BINS = ((-9, 0), (0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9))
for m in MET:
    ch_H, ch_T = KCL[m] * np.abs(xHd[m]) / SIG[m], KCL[m] * np.abs(xT[m]) / SIG[m]
    prof = " ".join(f"[{lo},{hi}) H {1e4 * np.mean((bH[m] >= lo) & (bH[m] < hi)):.2f} / T {1e4 * np.mean((bT[m] >= lo) & (bT[m] < hi)):.2f};"
                    for lo, hi in BINS[:-1])
    print(f"{m} rows per 10k by distance: {prof}")
    print(f"   change >0.01 sigma per 10k: H {1e4 * np.mean(ch_H > .01):.1f} test {1e4 * np.mean(ch_T > .01):.1f}; "
          f">0.1 sigma: H {1e4 * np.mean(ch_H > .1):.1f} test {1e4 * np.mean(ch_T > .1):.1f}; rms change/sigma H "
          f"{np.sqrt(np.mean(ch_H ** 2)):.5f} test {np.sqrt(np.mean(ch_T ** 2)):.5f}; mean x^2/sigma^2 H "
          f"{np.mean(xHd[m] ** 2) / SIG[m] ** 2:.3e} test {np.mean(xT[m] ** 2) / SIG[m] ** 2:.3e}")
    i = int(np.argmin(xT[m]))
    print(f"   largest test change: p {pT[m][i]:.1f} -> {min(pT[m][i] + KCL[m] * xT[m][i], GMAX if m == 'gas' else np.inf):.1f}")

# ---------------- C. projection
fz = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
kstar = {}
for m in MET:
    # P-OOF teacher (single fold model trained on 90% of P)
    pp = z[f"{m}_pP"]
    xP = mills_shift(pp, sdev(m, pp, MP), GMAX_P if m == "gas" else GMAX_P / AP)
    k_POOF = float(xP @ z[f"{m}_rP"] / (xP @ xP))
    # full-train OOF teacher (single fold model trained on 90% of all train rows)
    pf, yf = fz[f"{m}_p"], fz[f"{m}_y"]
    Atr, Mtr = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
    assert np.allclose(yf, tr[COL[m]].values, rtol=1e-12)
    xf = mills_shift(pf, sdev(m, pf, Mtr), GMAX if m == "gas" else GMAX / Atr)
    k_full = float(xf @ (yf - pf) / (xf @ xf))
    k_full_P = float(xf[P] @ (yf - pf)[P] / (xf[P] @ xf[P]))
    k_full_H = float(xf[H] @ (yf - pf)[H] / (xf[H] @ xf[H]))
    k_H = float(xHd[m] @ rH[m] / (xHd[m] @ xHd[m]))
    kstar[m] = dict(k_POOF=k_POOF, k_full=k_full, k_full_Prows=k_full_P, k_full_Hrows=k_full_H, k_Hcorr=k_H,
                    k_test_est=k_H * k_full / k_POOF)
    print(f"{m}: k* P-OOF {k_POOF:.4f} | full-train OOF {k_full:.4f} (P rows {k_full_P:.4f}, H rows {k_full_H:.4f}) | "
          f"H corrected {k_H:.4f} -> test regime estimate {k_H * k_full / k_POOF:.4f}")
proj = {}
for m in MET:
    nrH = float(np.sqrt(np.mean(rH[m] ** 2)) / SIG[m])
    k = KCL[m]

    def dN(kst, mx2):
        return (k * k - 2 * k * kst) * mx2 / (2 * nrH * SIG[m] ** 2)
    mx2H, mx2T = float(np.mean(xHd[m] ** 2)), float(np.mean(xT[m] ** 2))
    actual_H = float(np.sqrt(np.mean((rH[m] - k * xHd[m]) ** 2)) / SIG[m] - nrH)
    proj[m] = dict(H_formula_kH=dN(kstar[m]["k_Hcorr"], mx2H), H_actual_in_sample=actual_H,
                   test_same_kstar=dN(kstar[m]["k_Hcorr"], mx2T), test_scaled_kstar=dN(kstar[m]["k_test_est"], mx2T))
    # non-parametric: per-row H gain by distance bin x test bin share
    g = (rH[m] - k * xHd[m]) ** 2 - rH[m] ** 2
    dmse = 0.0
    for lo, hi in ((-9, -0.5), (-0.5, 0), (0, .25), (.25, .5), (.5, .75), (.75, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)):
        sh, stt = (bH[m] >= lo) & (bH[m] < hi), (bT[m] >= lo) & (bT[m] < hi)
        if sh.sum():
            dmse += g[sh].mean() * stt.mean()
    proj[m]["test_binwise"] = dmse / (2 * nrH * SIG[m] ** 2)
    print(f"{m}: dNRMSE formula on H {proj[m]['H_formula_kH']:+.7f} (actual in-sample {actual_H:+.7f}) | test, same k* "
          f"{proj[m]['test_same_kstar']:+.7f} | test, scaled k* {proj[m]['test_scaled_kstar']:+.7f} | test bin-wise "
          f"{proj[m]['test_binwise']:+.7f}")
for key in ("H_formula_kH", "test_same_kstar", "test_scaled_kstar", "test_binwise"):
    print(f"TOTAL {key}: {(proj['gas'][key] + proj['eui'][key]) / 5:+.8f}")
res["B_C"] = dict(kstar=kstar, proj=proj)
json.dump(res, open(f"{OUT}/l02_transfer.json", "w"), indent=1, default=float)
print(f"wrote {OUT}/l02_transfer.json; wall {time.time() - t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
