"""Honest check of a multiplicative small-meter fix for M=5 (and M=6) on top of the audit rules: delta fitted on P
(teacher residuals), applied to the corrected H predictions, scored on H. Also re-measures M<=3 and M=4 the same way."""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz")
P, H, M = z["P"], z["H"], z["M"]
MP, MH = M[P], M[H]
for grp in ((2, 3), (4, 4), (5, 5), (6, 6), (7, 8)):
    tot = 0.0
    msg = []
    for m in ("gas", "eui", "elec"):
        pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
        gP = (MP >= grp[0]) & (MP <= grp[1])
        gH = (MH >= grp[0]) & (MH <= grp[1])
        delta = float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
        y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
        q = p.copy()
        q[gH] *= 1 + delta
        d = (np.sqrt(np.mean((y - q) ** 2)) - np.sqrt(np.mean((y - p) ** 2))) / SIG[m]
        tot += d / 5
        msg.append(f"{m} delta {delta:+.4f} nH {gH.sum()} dNRMSE {d:+.7f}")
    print(f"M in {grp}: total {tot:+.7f} | " + " | ".join(msg), flush=True)
