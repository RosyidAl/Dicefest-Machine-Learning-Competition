"""Audit postproc 16: after the small-M shift (S1, P-OOF deltas), is there any calibration signal left?
5-fold cross-fitted on H: lower-bound truncated-normal term, lower-tail hinge, affine, M=5 group shift, E<=2 group
shift; and the M-by-p-quantile residual table again.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/16_after_s1.py
"""
import sys

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], Mall[P], Mall[H]
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in SIG}
GMAX, EUIMIN = yP["gas"].max(), yP["eui"].min()
EAMIN, EAMAX = (yP["elec"] / AP).min(), (yP["elec"] / AP).max()
folds = list(KFold(5, shuffle=True, random_state=0).split(AH))
tot = {}
for m in ("gas", "eui", "elec"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    y = z[f"{m}_y"]
    p = np.clip(z[f"{m}_corrected"], 0, None).copy()
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    a0, b0 = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]
    s = p * np.sqrt(np.clip(a0 + b0 / MH, 1e-8, None))
    lo = {"gas": EUIMIN * AH, "eui": np.full(len(AH), EUIMIN), "elec": EAMIN * AH}[m]
    hi = {"gas": np.full(len(AH), GMAX), "eui": GMAX / AH, "elec": EAMAX * AH}[m]
    a, bnd = (lo - p) / s, (hi - p) / s
    Z = np.clip(norm.cdf(bnd) - norm.cdf(a), 1e-300, None)
    t_lo, t_hi = s * norm.pdf(a) / Z, -s * norm.pdf(bnd) / Z
    zl = (p - lo) / s
    base = np.sqrt(np.mean((y - p) ** 2)) / SIG[m]
    print(f"== {m} after S1: {base:.6f}")
    X0 = t_hi[:, None]
    cands = {"tn_hi (S3)": X0,
             "S3 + tn_lo": np.column_stack([t_hi, t_lo]),
             "S3 + hinge lo": np.column_stack([t_hi] + [s * np.maximum(0, t - zl) for t in (1.5, 2.5, 3.5, 5)]),
             "S3 + affine": np.column_stack([t_hi, np.ones(len(p)), p]),
             "S3 + M=5 shift": np.column_stack([t_hi, p * (MH == 5)]),
             "S3 + M<=3 extra": np.column_stack([t_hi, p * (MH <= 3)])}
    s3 = None
    for name, X in cands.items():
        out = np.empty(len(p))
        for fa, fb in folds:
            beta = np.linalg.lstsq(X[fa], (y - p)[fa], rcond=None)[0]
            out[fb] = np.clip(p[fb] + X[fb] @ beta, lo[fb], hi[fb])
        d = np.sqrt(np.mean((y - out) ** 2)) / SIG[m] - base
        if s3 is None:
            s3 = d
        print(f"   {name:16s} d {d:+.7f} (vs S3 alone {d - s3:+.7f}, total {(d - s3) / 5:+.8f})")
        tot[name] = tot.get(name, 0) + (d - s3) / 5
print("extra over S3, TOTAL:", {k: round(v, 8) for k, v in tot.items()})
