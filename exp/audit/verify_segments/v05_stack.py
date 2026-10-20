"""Stacking risk: the same cap effect was also found by other lenses (exp/audit/tried 'G' = linear shrink above 9e5 with
a 1e6 clip, beta fitted on the P teacher; exp/audit/postproc S3/S4 = soft cap + hard clip). Score on H corrected (gas and
EUI via eui*A): IMR alone, G alone, IMR followed by G (each with its own P-fitted parameter, i.e. a naive stack).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/v05_stack.py \
       > exp/audit/verify_segments/v05_stack.log
"""
import numpy as np
from scipy.special import log_ndtr

SG, SE = 151750.04, 40.65
U, K, LO, KNOT = 1e6, 0.08, 6e5, 9e5
LOG2PI = 0.5 * np.log(2 * np.pi)


def imr(g):
    g = np.asarray(g, float)
    out = g.copy()
    m = g > LO
    s = K * g[m]
    a = (U - g[m]) / s
    out[m] = g[m] - s * np.exp(-0.5 * a * a - LOG2PI - log_ndtr(a))
    return out


z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
yP, pP = z["gas_pP"] + z["gas_rP"], z["gas_pP"]
hP = np.maximum(pP - KNOT, 0)
beta = float(hP @ (yP - pP) / (hP @ hP))
G = lambda g: np.minimum(g + beta * np.maximum(g - KNOT, 0), U)  # noqa: E731
yH, pc = z["gas_y"], z["gas_corrected"]
yE, pE = z["eui_y"], z["eui_corrected"]
gE = pE * A[H]


def score(f):
    dg = (np.sqrt(np.mean((yH - f(pc)) ** 2)) - np.sqrt(np.mean((yH - pc) ** 2))) / SG / 5
    pn = np.where(gE > 0, pE * f(gE) / np.where(gE > 0, gE, 1.0), pE)
    de = (np.sqrt(np.mean((yE - pn) ** 2)) - np.sqrt(np.mean((yE - pE) ** 2))) / SE / 5
    return dg, de


print(f"G beta fitted on P teacher: {beta:.4f}")
for name, f in [("IMR k=0.08 alone", imr), ("G alone", G), ("IMR then G (naive stack)", lambda g: G(imr(g))),
                ("IMR then clip at train max", lambda g: np.minimum(imr(g), 999764.9))]:
    dg, de = score(f)
    print(f"  {name:28s} gas {dg:+.8f}  eui {de:+.8f}  total {dg + de:+.8f}")
