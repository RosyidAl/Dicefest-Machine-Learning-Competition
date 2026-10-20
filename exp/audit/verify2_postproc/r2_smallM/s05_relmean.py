"""Claim numbers (2)/(3): mean relative residual r/p for M == 3 on P (P-OOF teacher) and on H (corrected).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/r2_smallM/s05_relmean.py"""
import numpy as np

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M = z["P"], z["H"], z["M"]
for m in ("gas", "eui", "elec"):
    sP, sH = M[P] == 3, M[H] == 3
    q = z[m + "_rP"][sP] / z[m + "_pP"][sP]
    qh = (z[m + "_y"][sH] - z[m + "_corrected"][sH]) / z[m + "_corrected"][sH]
    print(m, "P mean r/p %+.3f (se %.3f)" % (q.mean(), q.std(ddof=1) / np.sqrt(len(q))),
          "| H corr mean r/p %+.3f (se %.3f)" % (qh.mean(), qh.std(ddof=1) / np.sqrt(len(qh))))
