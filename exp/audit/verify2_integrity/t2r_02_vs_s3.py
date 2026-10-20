"""verify2_integrity t2r_02: does the integrity-lens truncation shrink add anything over the S3 soft cap already
adopted in D015 (sub_v19ncapp / sub_v19nca2pp = BEST_CURRENT)? Own re-implementation of the deployed S1 + S3 rule
(exp/audit/postproc_nca/13_candidate_nca.py is only read for its definition, not imported), scored on H.

  base1  = clip(corrected, 0) with S1 (small-M multiplicative shift, deltas from P: LS bias of pP for M in 2..3, M == 4)
  S3     = base1 + kS3 * TNshift(base1, s = p sqrt(a + b/M), U = max P gas)   (k = 1 shape, kS3 LS-fitted)
           kS3 'deployed' = fitted on all of H (in-sample, as in D015), 'cf' = 5-fold x 10 seeds cross-fitted on H
  CLAIM  = TNshift with s -> 0.7 s, U = 1e6, c from P (integrity parameters, re-derived in t2r_01)
Comparisons (total-score units, gas + EUI):
  claim_replacing_S3  : base1 + CLAIM          vs S3 (cf)
  claim_on_top_of_S3  : S3 + CLAIM(S3 output)  vs S3 (deployed / cf)
  refit_on_top_of_S3  : S3(cf) + c * shift_k0.7 with c cross-fitted on H (does any residual signal remain?)
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/t2r_02_vs_s3.py
"""
import json

import numpy as np
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

OUT = "exp/audit/verify2_integrity/t2r_02_vs_s3.json"
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


def nrmse(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def dtot(y, pn, p0, m):
    return (nrmse(y, pn, m) - nrmse(y, p0, m)) / 5


def claim_x(m, p):
    """integrity-claim shift column (k, c from P) on the metric scale"""
    k, _ = KC[m]
    pg = p * AH if m == "eui" else p
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(ABG[0] + ABG[1] / MH, 1e-6)), 1.0)
    x = shift(pg, s, 1e6)
    return x / AH if m == "eui" else x


def shape_x(m, p, k, U):
    pg = p * AH if m == "eui" else p
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(ABG[0] + ABG[1] / MH, 1e-6)), 1.0)
    x = shift(pg, s, U)
    return x / AH if m == "eui" else x


def ls(x, r):
    return float(x @ r / (x @ x))


res = {}
SEEDS = range(10)
for m in ("gas", "eui"):
    y = z[f"{m}_y"]
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    p0 = np.clip(z[f"{m}_corrected"], 0, None)
    # S1 (deltas from P)
    b1 = p0.copy()
    deltas = {}
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        dl = float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
        deltas[f"M{lo}-{hi}"] = dl
        b1[gH] *= 1 + dl
    # S3 variance model as deployed: per metric, (r/p)^2 = a + b/M on all P rows
    sd = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]
    s3s = b1 * np.sqrt(np.clip(sd[0] + sd[1] / MH, 1e-8, None))
    x3 = shift(b1, s3s, GMAX_P if m == "gas" else GMAX_P / AH)
    k_dep = ls(x3, y - b1)
    s3_dep = b1 + k_dep * x3
    # S3 cross-fitted (honest) and claim-refit on top, per seed
    out = dict(deltas_S1=deltas, var_S3=sd.tolist(), kS3_allH=k_dep, base0=nrmse(y, p0, m), base1=nrmse(y, b1, m),
               S1_vs_base0=dtot(y, b1, p0, m), S3dep_vs_base1_insample=dtot(y, s3_dep, b1, m))
    cf, top_cf, refit_top, k07_cf, claim_top_dep = [], [], [], [], []
    for seed in SEEDS:
        s3_cf = b1.copy()
        k07 = b1.copy()
        for a, b in KFold(5, shuffle=True, random_state=200 + seed).split(H):
            s3_cf[b] = b1[b] + ls(x3[a], (y - b1)[a]) * x3[b]
            xk = shape_x(m, b1, 0.7, 1e6)
            k07[b] = b1[b] + ls(xk[a], (y - b1)[a]) * xk[b]
        cf.append(dtot(y, s3_cf, b1, m))
        k07_cf.append(dtot(y, k07, b1, m))
        # claim (P-fitted) applied on top of the honest S3 output
        top = s3_cf + KC[m][1] * claim_x(m, s3_cf)
        top_cf.append(dtot(y, top, s3_cf, m))
        # refit: c for the k=0.7 shape on top of S3(cf), cross-fitted with a different split
        xr = shape_x(m, s3_cf, 0.7, 1e6)
        rf = s3_cf.copy()
        for a, b in KFold(5, shuffle=True, random_state=300 + seed).split(H):
            rf[b] = s3_cf[b] + ls(xr[a], (y - s3_cf)[a]) * xr[b]
        refit_top.append(dtot(y, rf, s3_cf, m))
    claim_rep = b1 + KC[m][1] * claim_x(m, b1)
    claim_top_dep = s3_dep + KC[m][1] * claim_x(m, s3_dep)
    out.update(
        S3cf_vs_base1=float(np.mean(cf)), S3cf_sd_seeds=float(np.std(cf)),
        k07cf_vs_base1=float(np.mean(k07_cf)),
        claim_vs_base1=dtot(y, claim_rep, b1, m),
        claim_replacing_S3cf=float(dtot(y, claim_rep, b1, m) - np.mean(cf)),
        claim_on_top_of_S3dep=dtot(y, claim_top_dep, s3_dep, m),
        claim_on_top_of_S3cf=float(np.mean(top_cf)),
        refit_k07_on_top_of_S3cf=float(np.mean(refit_top)),
        max_pred_after={"base1": float((b1 * (AH if m == "eui" else 1)).max()),
                        "S3dep": float((s3_dep * (AH if m == "eui" else 1)).max()),
                        "claim": float((claim_rep * (AH if m == "eui" else 1)).max()),
                        "claim_on_S3dep": float((claim_top_dep * (AH if m == "eui" else 1)).max())})
    res[m] = out
    print(m, json.dumps({k: (round(v * 1e5, 3) if isinstance(v, float) and abs(v) < 1e-2 else v)
                         for k, v in out.items()}, default=float))
tot = {k: res["gas"][k] + res["eui"][k] for k in ("S1_vs_base0", "S3dep_vs_base1_insample", "S3cf_vs_base1",
                                                  "k07cf_vs_base1", "claim_vs_base1", "claim_replacing_S3cf",
                                                  "claim_on_top_of_S3dep", "claim_on_top_of_S3cf",
                                                  "refit_k07_on_top_of_S3cf")}
res["total"] = tot
print("TOTAL (e-5):", {k: round(v * 1e5, 3) for k, v in tot.items()})
json.dump(res, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
