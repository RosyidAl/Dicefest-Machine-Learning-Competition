"""How noisy is the public LB (30% of test) as a proxy for the private LB (70%)? Using H (128,791 rows, honest
corrected predictions) as a stand-in for a test set: random 30/70 splits, sd of (public - private) total score for
gas + EUI + elec (/5; clean ratio is exact, adoption omitted), and for the difference between two candidates
(corrected vs corrected + G+S fixes, i.e. how reliably the public LB ranks a ~-0.00007 change)."""
import json

import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AH, MH = A[H], M[H]
prm = json.load(open("exp/audit/tried/combo_params_P.json"))
beta, KNOT, CAP = prm["beta"], prm["knot"], prm["cap"]
fac = prm["factors"]
e0, e1 = {}, {}
for m in SIG:
    y, c = z[f"{m}_y"], z[f"{m}_corrected"]
    c2 = c.copy()
    for k, sel in (("<=3", MH <= 3), ("4", MH == 4)):
        c2[sel] *= 1 + 0.5 * (fac[m][k] - 1)  # S with shrink 0.5
    if m == "gas":
        c2 = np.minimum(c2 + beta * np.maximum(c2 - KNOT, 0), CAP)
    elif m == "eui":
        g = c2 * AH
        c2 = np.where(g > KNOT, np.minimum(g + beta * (g - KNOT), CAP) / AH, c2)
    e0[m], e1[m] = (y - c) ** 2, (y - c2) ** 2
rng = np.random.default_rng(5)
n = len(H)
shift, dpub, dpriv = [], [], []
for _ in range(500):
    pub = np.zeros(n, bool)
    pub[rng.choice(n, int(0.3 * n), replace=False)] = True
    s0 = lambda e, msk: sum(np.sqrt(e[m][msk].mean()) / SIG[m] for m in SIG) / 5  # noqa: E731
    shift.append(s0(e0, pub) - s0(e0, ~pub))
    dpub.append(s0(e1, pub) - s0(e0, pub))
    dpriv.append(s0(e1, ~pub) - s0(e0, ~pub))
shift, dpub, dpriv = map(np.array, (shift, dpub, dpriv))
print(f"public - private total (3 metrics/5): mean {shift.mean():+.6f} sd {shift.std():.6f}")
print(f"G+S(shrink .5) change: public mean {dpub.mean():+.7f} sd {dpub.std():.7f}; private mean {dpriv.mean():+.7f} sd {dpriv.std():.7f}; "
      f"P(public<0) {(dpub < 0).mean():.3f} P(private<0) {(dpriv < 0).mean():.3f}; corr(pub, priv) {np.corrcoef(dpub, dpriv)[0, 1]:+.2f}")
