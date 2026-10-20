"""Freeze the post-processing parameters of sub_v19nca2pp (exactly as 13_candidate_nca2.py computes them) into the
resid_knn json of exp/v19nca2pp: S1 deltas (full-train VC OOF, 12_fullvc_smallM_nca2.json), S3 noise model (a0, b0
from P-OOF residuals) and k (fitted on H after S1). Bounds are recomputed from train inside the pipeline."""
import json

import numpy as np
from scipy.stats import norm

MET = ("gas", "eui", "elec")
fo = json.load(open("exp/audit/postproc_nca2/12_fullvc_smallM_nca2.json"))
delta = {m: {"M23": fo[m]["M<=3"]["ls_delta"], "M4": fo[m]["M=4"]["ls_delta"]} for m in MET}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
MP, MH, AH = Mall[P], Mall[H], A[H]
SD = {}
for m in MET:
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(p)), 1 / MP]), (r / p) ** 2, rcond=None)[0]
GMAX_P = (z["gas_pP"] + z["gas_rP"]).max()


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


kS3 = {}
for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    p = np.clip(z[f"{m}_corrected"], 0, None).copy()
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / MH, 1e-8, None))
    x = tn_hi(p, s, GMAX_P if m == "gas" else GMAX_P / AH)
    kS3[m] = float(x @ (z[f"{m}_y"] - p) / (x @ x))
pp = {"s1": delta, "s3": {m: {"k": kS3[m], "a0": float(SD[m][0]), "b0": float(SD[m][1])} for m in ("gas", "eui")}}
j = json.load(open("exp/resid_knn_vc_AMp_e3.json"))
j["postproc"] = pp
json.dump(j, open("exp/v19nca2pp/resid_knn.json", "w"), indent=1)
print(json.dumps(pp))
