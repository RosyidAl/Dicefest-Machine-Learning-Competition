"""Audit postproc 08: gas = EUI * A consistency of the two separately blended + corrected predictions on H.

gas' = gas_c + w * (eui_c * A - gas_c),  eui' = eui_c + v * (gas_c / A - eui_c);  w, v by least squares, 5-fold
cross-fitted on H; variants: global weight, weight piecewise linear in the rank of A (hat basis, 2 and 3 knots),
forced consistency (w = 1 / v = 1) and the A^2-weighted / unweighted geometric compromise.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/08_consistency.py
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH = A[H]
yg, ye = z["gas_y"], z["eui_y"]
gc, ec = np.clip(z["gas_corrected"], 0, None), np.clip(z["eui_corrected"], 0, None)
print("rms (gas_c - eui_c*A)/sigma_gas", float(np.sqrt(np.mean((gc - ec * AH) ** 2)) / SIG["gas"]),
      "corr of the two gas errors", float(np.corrcoef(yg - gc, yg - ec * AH)[0, 1]))


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def hat(x, k):
    """hat basis on [0,1] with k knots (k>=2)"""
    t = np.linspace(0, 1, k)
    return np.column_stack([np.clip(1 - np.abs(x - tj) * (k - 1), 0, None) for tj in t])


rankA = (np.argsort(np.argsort(AH)) + 0.5) / len(AH)
folds = list(KFold(5, shuffle=True, random_state=0).split(AH))
tot = {}
for m, y, p, other in (("gas", yg, gc, ec * AH), ("eui", ye, ec, gc / AH)):
    base = nr(y, p, m)
    dlt = other - p
    print(f"== {m} base {base:.6f}; forced (w=1) d {nr(y, other, m) - base:+.7f}; w=0.5 d {nr(y, p + 0.5 * dlt, m) - base:+.7f}")
    for name, B in (("global", np.ones((len(p), 1))), ("rankA hat2", hat(rankA, 2)), ("rankA hat3", hat(rankA, 3)),
                    ("rankA hat5", hat(rankA, 5))):
        X = B * dlt[:, None]
        out = np.empty(len(p))
        ws = []
        for a, b in folds:
            beta = np.linalg.lstsq(X[a], (y - p)[a], rcond=None)[0]
            ws.append(np.round(beta, 3).tolist())
            out[b] = np.clip(p[b] + X[b] @ beta, 0, None)
        d = nr(y, out, m) - base
        tot[name] = tot.get(name, 0) + d / 5
        print(f"   {name:11s} cross-fitted d {d:+.7f} total {d / 5:+.8f} weights fold0 {ws[0]}")
print("TOTAL", {k: round(v, 8) for k, v in tot.items()})
