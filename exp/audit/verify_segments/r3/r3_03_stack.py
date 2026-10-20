"""verify_segments r3: overlap of the claimed IMR cap calibration with the soft cap S3 that D015 already adopted
(exp/audit/postproc_nca/13_candidate_nca.py: p += kS3 * tn_hi(p, s), s = p sqrt(a0 + b0 / M), upper bound = max gas).

1. Submissions: is S3 already inside BEST_CURRENT (sub_v19nca2pp vs its pre-post-processing sub_v19nca2)?
   What would the claimed calibration still change if applied on top of BEST_CURRENT?
2. H chain (honest): S3 alone (kS3 fitted on the P teacher, and 5-fold cross-fitted on H), the claimed IMR alone,
   and the stack S3 -> IMR (IMR k fitted on the P teacher after S3, or the fixed claimed k 0.08).
   The increment of the stack over S3 alone = what the claim would add to the current pipeline.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/r3/r3_03_stack.py \
         > exp/audit/verify_segments/r3/r3_03_stack.log 2>&1
"""
import resource
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr
from scipy.stats import norm

T0 = time.time()
SIG = {"gas": 151750.04, "eui": 40.65}
CAP, THR = 1e6, 6e5
L2PI = 0.5 * np.log(2 * np.pi)
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"


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


# ------------------------------------------------------------------ 1. submissions
cols = ["id", GAS, EUI]
s0 = pd.read_csv("submissions/sub_v19nca2.csv", usecols=cols, float_precision="round_trip")
s1 = pd.read_csv("submissions/sub_v19nca2pp.csv", usecols=cols, float_precision="round_trip")
bc = pd.read_csv("submissions/BEST_CURRENT.csv", usecols=cols, float_precision="round_trip")
vc = pd.read_csv("submissions/sub_v19vce.csv", usecols=cols, float_precision="round_trip")
At = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06"])["bld_dim_06"].values.astype(float)
print(f"BEST_CURRENT == sub_v19nca2pp (gas, eui): {np.array_equal(bc[GAS].values, s1[GAS].values)} "
      f"{np.array_equal(bc[EUI].values, s1[EUI].values)}")
for nm, s in (("sub_v19vce", vc), ("sub_v19nca2", s0), ("sub_v19nca2pp", s1)):
    g = s[GAS].values
    print(f"  {nm:14s} gas max {g.max():10.0f}  rows > 1e6 {int((g > CAP).sum())}  rows > 0.95e6 {int((g > 0.95e6).sum())}"
          f"  eui*A max {np.max(s[EUI].values * At):10.0f}")
g0, g1 = s0[GAS].values, s1[GAS].values
EDG = np.array([0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 9.0]) * 1e6
b = np.digitize(g0, EDG) - 1
print("mean change of gas pp - pre (by pre level) vs what the claimed IMR k 0.08 would do to the pre file:")
for j in range(len(EDG) - 1):
    m = b == j
    if m.any():
        print(f"   [{EDG[j] / 1e6:.2f},{EDG[j + 1] / 1e6:.2f}) n {m.sum():4d}  D015 pp {np.mean(g1[m] - g0[m]):+9.0f}  "
              f"IMR(pre) {np.mean(cal(g0[m], 0.08) - g0[m]):+9.0f}  IMR on top of pp {np.mean(cal(g1[m], 0.08) - g1[m]):+9.0f}")
d_top = cal(g1, 0.08) - g1
print(f"IMR on top of BEST_CURRENT: gas rows changed by > 0.1 sigma {int((np.abs(d_top) > 0.1 * SIG['gas']).sum())}, "
      f"max |change| {np.abs(d_top).max():.0f}, rms change/sigma {np.sqrt(np.mean(d_top ** 2)) / SIG['gas']:.5f}")

# ------------------------------------------------------------------ 2. H chain
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
MP, MH, AH, AP = M[P], M[H], A[H], A[P]
GMAX_P = float((z["gas_pP"] + z["gas_rP"]).max())
SD = {}
for m in ("gas", "eui"):
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(p)), 1 / MP]), (r / p) ** 2, rcond=None)[0]
print(f"\nS3 noise model (r/p)^2 = a0 + b0/M on P: gas {np.round(SD['gas'], 5)} eui {np.round(SD['eui'], 5)}; "
      f"upper bound (P max gas) {GMAX_P:.1f}")


def tn_x(p, Mv, Av, m):
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / Mv, 1e-8, None))
    hi = GMAX_P if m == "gas" else GMAX_P / Av
    bb = (hi - p) / s
    return -s * norm.pdf(bb) / np.clip(norm.cdf(bb), 1e-300, None)


def nr(e, m):
    return float(np.sqrt(np.mean(e * e)) / SIG[m])


def ls_k(x, r):
    return float(x @ r / (x @ x))


def fit_imr(y, p, kind, Av=None):
    if kind == "gas":
        mm = p > THR
        f = lambda k: float(np.sum((y[mm] - cal(p[mm], k)) ** 2))  # noqa: E731
    else:
        mm = p * Av > THR
        f = lambda k: float(np.sum((y[mm] - cal_eui(p[mm], Av[mm], k)) ** 2))  # noqa: E731
    grid = np.arange(0.005, 0.3001, 0.0025)
    k0 = grid[int(np.argmin([f(k) for k in grid]))]
    return float(minimize_scalar(f, bounds=(max(0.001, k0 - 0.003), k0 + 0.003), method="bounded").x)


for m in ("gas", "eui"):
    yP = z[f"{m}_pP"] + z[f"{m}_rP"]
    pP = np.clip(z[f"{m}_pP"], 0, None)
    yH, pH = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    base = nr(yH - pH, m)
    xP, xH = tn_x(pP, MP, AP, m), tn_x(pH, MH, AH, m)
    # S3 with kS3 fitted on the P teacher (honest for H)
    k3P = ls_k(xP, yP - pP)
    s3 = pH + k3P * xH
    # S3 with kS3 fitted on all of H (as adopted in D015; in-sample on H, diagnostic only)
    k3H = ls_k(xH, yH - pH)
    # S3 cross-fitted on H (5-fold)
    s3cf = pH.copy()
    perm = np.random.default_rng(0).permutation(len(H))
    for part in np.array_split(perm, 5):
        tr_ = np.setdiff1d(perm, part)
        s3cf[part] = pH[part] + ls_k(xH[tr_], yH[tr_] - pH[tr_]) * xH[part]
    # claimed IMR alone (k fitted on P teacher in gas units for gas; for eui the gas-scale k as claimed)
    kI = fit_imr(z["gas_pP"] + z["gas_rP"], np.clip(z["gas_pP"], 0, None), "gas")
    imr = cal(pH, kI) if m == "gas" else cal_eui(pH, AH, kI)
    imr08 = cal(pH, 0.08) if m == "gas" else cal_eui(pH, AH, 0.08)
    # stack: S3 then IMR, IMR k refit on the P teacher after S3 (honest) and the fixed claimed 0.08
    s3P = pP + k3P * xP
    kS = fit_imr(yP, s3P, "gas") if m == "gas" else fit_imr(yP, s3P, "eui", AP)
    st = cal(s3, kS) if m == "gas" else cal_eui(s3, AH, kS)
    st08 = cal(s3, 0.08) if m == "gas" else cal_eui(s3, AH, 0.08)
    d = lambda p: (nr(yH - p, m) - base) / 5  # noqa: E731
    print(f"\n[{m}] H corrected base {base:.6f}; kS3 P-fit {k3P:.4f} all-H {k3H:.4f}; IMR k P-fit {kI:.4f}; "
          f"IMR k refit after S3 {kS:.4f}")
    print(f"   S3 alone (P-fit k)            dTotal {d(s3):+.8f}")
    print(f"   S3 alone (5-fold H cross-fit)  dTotal {d(s3cf):+.8f}")
    print(f"   S3 alone (all-H k, in-sample)  dTotal {d(pH + k3H * xH):+.8f}")
    print(f"   claimed IMR alone (P-fit k)    dTotal {d(imr):+.8f}   (k 0.08: {d(imr08):+.8f})")
    print(f"   stack S3 -> IMR (refit k)      dTotal {d(st):+.8f}   increment over S3 {d(st) - d(s3):+.8f}")
    print(f"   stack S3 -> IMR (k 0.08)       dTotal {d(st08):+.8f}   increment over S3 {d(st08) - d(s3):+.8f}")
    print(f"   corr of the two adjustments on rows > 0.85e6 (gas units): "
          f"{np.corrcoef((s3 - pH)[(pH * (AH if m == 'eui' else 1)) > 0.85e6], (imr - pH)[(pH * (AH if m == 'eui' else 1)) > 0.85e6])[0, 1]:.3f}")
print(f"\nelapsed {time.time() - T0:.1f} s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
