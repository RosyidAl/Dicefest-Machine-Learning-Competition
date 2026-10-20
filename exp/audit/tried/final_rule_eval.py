"""Exact candidate rule (as make_candidate.py: S with shrink 0.5, G with P-fitted beta) on the honest H chain, and the
same rule with S shrink 0.75 / 1.0; bootstrap over H rows (private-LB size)."""
import json

import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, A, M = z["H"], z["A"], z["M"]
AH, MH = A[H], M[H]
prm = json.load(open("exp/audit/tried/combo_params_P.json"))
beta, KNOT, CAP, fac = prm["beta"], prm["knot"], prm["cap"], prm["factors"]
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(H), len(H)) for _ in range(500)]


def rule(m, c, shrink):
    v = c.copy()
    for k, sel in (("<=3", MH <= 3), ("4", MH == 4)):
        v[sel] *= 1 + shrink * (fac[m][k] - 1)
    if m == "gas":
        s = v > KNOT
        v[s] = np.minimum(v[s] + beta * (v[s] - KNOT), CAP)
    elif m == "eui":
        g = v * AH
        s = g > KNOT
        v[s] = np.minimum(g[s] + beta * (g[s] - KNOT), CAP) / AH[s]
    return v


for shrink in (0.5, 0.75, 1.0):
    tot, tb = 0.0, 0.0
    msg = []
    for m in SIG:
        y, c = z[f"{m}_y"], z[f"{m}_corrected"]
        e0, e1 = (y - c) ** 2, (y - rule(m, c, shrink)) ** 2
        dd = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m]
        tot += dd / 5
        tb = tb + np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] / 5 for b in BOOT])
        msg.append(f"{m} {np.sqrt(e1.mean()) / SIG[m]:.6f} ({dd:+.7f})")
    print(f"S shrink {shrink}: {'; '.join(msg)} | TOTAL {tot:+.8f} sd {np.std(tb):.8f} P(<0) {(tb < 0).mean():.3f} "
          f"5%-95% [{np.quantile(tb, 0.05):+.7f}, {np.quantile(tb, 0.95):+.7f}]", flush=True)
