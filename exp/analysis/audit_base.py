"""Shared inputs for the 20 Oct final audit (exp/audit/): holdout-chain predictions on H for gas / eui / elec
(teacher = VC blend A2xM2xp2 of all hold members fitted on P-OOF; corrected = teacher + group/kNN correction with the
current configs, lambda 5-fold cross-fitted on H, i.e. the honest chain prediction behind sub_v19vce), the correction
feature matrices, and P-side fitted predictions / residuals. Light enough for analysis agents (no blend refit needed).

usage: python exp/analysis/audit_base.py
"""
import json
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
sys.path.insert(0, "exp/analysis")
from corr_est import chain  # noqa: E402
from select_corr import CACHE  # noqa: E402

tr, te, d, P, H, A, res = chain()
cfgs = json.load(open("exp/resid_knn_vc_AMp_e.json"))["configs"]
out = dict(P=P, H=H, A=A, M=tr["bld_dim_12"].values.astype(float), subn=d.subn)
for m in ("gas", "eui", "elec"):
    R = res[m]
    F = np.column_stack([R["G"]] + [np.load(f"{CACHE}/{m}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in cfgs[m]])
    pc = np.zeros(len(H))
    for a, b in KFold(5, shuffle=True, random_state=0).split(F):
        pc[b] = F[b] @ np.linalg.lstsq(F[a], R["rH"][a], rcond=None)[0]
    y = R["rH"] + R["pH"]
    out.update({f"{m}_y": y, f"{m}_teacher": R["pH"], f"{m}_corrected": R["pH"] + pc, f"{m}_F": F.astype(np.float32),
                f"{m}_pP": R["pP"], f"{m}_rP": R["rP"]})
    print(m, "teacher", round(float(np.sqrt(np.mean(R["rH"] ** 2)) / R["sig"]), 5), "corrected",
          round(float(np.sqrt(np.mean((y - R["pH"] - pc) ** 2)) / R["sig"]), 5), flush=True)
np.savez_compressed("exp/audit/chain_H.npz", **out)
print("wrote exp/audit/chain_H.npz")
