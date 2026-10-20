"""Audit postproc 17: 5-fold cross-fitted H effect of the recommended chain S1 (M 2-3 and M 4, P-OOF deltas) + S3 + S4
(no S2), per metric and total, reusing 11_combo.chain. Also per-fold totals.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/17_final_eval.py
"""
import runpy
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.argv = ["11_combo.py"]
src = open("exp/audit/postproc/11_combo.py").read().split("ALL = (")[0]  # definitions only, skip the analysis loop
ns = {}
exec(compile(src, "11_combo_defs", "exec"), ns)
chain, nr, MET, H, AH = ns["chain"], ns["nr"], ns["MET"], ns["H"], ns["AH"]
groups = [(2, 3), (4, 4)]
steps = ("S1", "S3", "S4")
pred = {s: {m: np.empty(len(H)) for m in MET} for s in ("S0", "S4")}
fold_tot = []
for a, b in KFold(5, shuffle=True, random_state=0).split(AH):
    r = chain(a, b, groups, steps)
    for s in pred:
        for m in MET:
            pred[s][m][b] = r[s][m]
    fold_tot.append(sum(nr(m, b, r["S4"][m]) - nr(m, b, r["S0"][m]) for m in MET) / 5)
idx = np.arange(len(H))
d = {m: nr(m, idx, pred["S4"][m]) - nr(m, idx, pred["S0"][m]) for m in MET}
print("S1+S3+S4 5-fold:", {m: round(v, 7) for m, v in d.items()}, "total", round(sum(d.values()) / 5, 8))
print("per-fold totals", np.round(fold_tot, 8))
