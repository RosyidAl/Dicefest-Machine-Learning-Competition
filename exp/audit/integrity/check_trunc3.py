"""Integrity audit 14: truncation adjustment with spread multiplier k (s -> k*s) and scale c, both fitted on P
(teacher P-OOF) by least squares; scored on H (honest chain). EUI: own c on P with the same k grid.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_trunc3.py
"""
import json
import numpy as np
from scipy.stats import norm
SIG = dict(gas=151750.04, eui=40.65)
p2 = json.load(open("exp/audit/integrity/check_trunc2.json"))
a0, b0 = p2["var_normal"]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
def adj(pg, m, k):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / m, 1e-6)), 1.0)
    a = (1e6 - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))
def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])
out = {}
for m in ("gas", "eui"):
    scP = A[P] if m == "eui" else np.ones(len(P)); scH = A[H] if m == "eui" else np.ones(len(H))
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    grid = {}
    for k in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        aP = adj(pP * scP, M[P], k) / scP
        c = float(np.sum(aP * rP) / np.sum(aP ** 2))
        sseP = float(np.sum((rP - c * aP) ** 2) - np.sum(rP ** 2))
        aH = adj(p * scH, M[H], k) / scH
        pa = p + c * aH
        grid[k] = dict(c=c, dSSE_P=sseP, H_delta_total=(nr(y, pa, m) - nr(y, p, m)) / 5,
                       H_halves=[(nr(y[i::2], pa[i::2], m) - nr(y[i::2], p[i::2], m)) / 5 for i in range(2)])
    kbest = min(grid, key=lambda k: grid[k]["dSSE_P"])
    out[m] = dict(grid=grid, k_best_on_P=kbest, H_delta_total_at_k_best=grid[kbest]["H_delta_total"])
out["sum_at_P_best"] = out["gas"]["H_delta_total_at_k_best"] + out["eui"]["H_delta_total_at_k_best"]
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_trunc3.json", "w"), indent=1)
