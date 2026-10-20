"""verify2_integrity t2r_04: paired uncertainty of 'claim form (k=0.7, c from P) replacing the adopted S3 soft cap
(k=1 shape)' on H, after S1. Same constructions as t2r_02 (own code). Paired row bootstrap (2000) and the 4 contiguous
sub_n blocks of H.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/t2r_04_paired.py
"""
import json

import numpy as np
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
p1 = json.load(open("exp/audit/verify2_integrity/t2r_01_repro.json"))
ABG = p1["var_model"]["gas"]
KC = {m: (p1["A_fitP_scoreH"][m]["k_P"], p1["A_fitP_scoreH"][m]["c_P"]) for m in ("gas", "eui")}
GMAX_P = float((z["gas_pP"] + z["gas_rP"]).max())


def shift(pg, s, U):
    a = (U - pg) / s
    return -s * np.exp(-0.5 * a * a - 0.5 * np.log(2 * np.pi) - log_ndtr(a))


Y, B1, S3D, S3C, CL = {}, {}, {}, {}, {}
for m in ("gas", "eui"):
    y = z[f"{m}_y"]
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    b1 = np.clip(z[f"{m}_corrected"], 0, None)
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        b1[gH] *= 1 + float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    sd = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]
    x3 = shift(b1, b1 * np.sqrt(np.clip(sd[0] + sd[1] / MH, 1e-8, None)), GMAX_P if m == "gas" else GMAX_P / AH)
    r = y - b1
    s3d = b1 + float(x3 @ r / (x3 @ x3)) * x3
    s3c = b1.copy()
    for a, b in KFold(5, shuffle=True, random_state=200).split(H):
        s3c[b] = b1[b] + float(x3[a] @ r[a] / (x3[a] @ x3[a])) * x3[b]
    pg = b1 * AH if m == "eui" else b1
    s = np.maximum(KC[m][0] * np.abs(pg) * np.sqrt(np.maximum(ABG[0] + ABG[1] / MH, 1e-6)), 1.0)
    xc = shift(pg, s, 1e6)
    xc = xc / AH if m == "eui" else xc
    Y[m], B1[m], S3D[m], S3C[m], CL[m] = y, b1, s3d, s3c, b1 + KC[m][1] * xc


def total(idx, Q, R):
    t = 0.0
    for m in ("gas", "eui"):
        y = Y[m][idx]
        t += (np.sqrt(np.mean((y - Q[m][idx]) ** 2)) - np.sqrt(np.mean((y - R[m][idx]) ** 2))) / SIG[m] / 5
    return float(t)


allidx = np.arange(len(H))
res = {"claim_minus_S3dep": total(allidx, CL, S3D), "claim_minus_S3cf": total(allidx, CL, S3C),
       "claim_minus_base1": total(allidx, CL, B1), "S3cf_minus_base1": total(allidx, S3C, B1)}
rng = np.random.default_rng(7)
bd, bc = [], []
for _ in range(2000):
    i = rng.integers(0, len(H), len(H))
    bd.append(total(i, CL, S3D))
    bc.append(total(i, CL, S3C))
bd, bc = np.array(bd), np.array(bc)
res["boot_claim_minus_S3dep"] = dict(sd=float(bd.std()), p_ge_0=float(np.mean(bd >= 0)))
res["boot_claim_minus_S3cf"] = dict(sd=float(bc.std()), p_ge_0=float(np.mean(bc >= 0)))
subnH = z["subn"][H]
q = np.quantile(subnH, [0.25, 0.5, 0.75])
blk = np.searchsorted(q, subnH, side="right")
res["blocks_claim_minus_S3cf_e5"] = [round(total(np.where(blk == g)[0], CL, S3C) * 1e5, 3) for g in range(4)]
res["blocks_claim_minus_base1_e5"] = [round(total(np.where(blk == g)[0], CL, B1) * 1e5, 3) for g in range(4)]
print(json.dumps({k: (round(v * 1e5, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
json.dump(res, open("exp/audit/verify2_integrity/t2r_04_paired.json", "w"), indent=1)
