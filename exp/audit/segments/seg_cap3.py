"""Gas high-end calibration: free piecewise-linear shift f(p / 1e6) (zero below the first knot) vs the one-parameter
truncated-normal (IMR) calibration, and IMR + piecewise shoulder. Honest: fit on the P-side teacher (OOF VC blend),
score on H corrected; and 5-fold x 10 cross-fit on H corrected.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_cap3.py > exp/audit/segments/seg_cap3.log
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold

SIG = 151750.04
U = 1e6
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H = z["P"], z["H"]
gas = pd.read_parquet("cache/train.parquet", columns=["target_macro_gas_kwh"])["target_macro_gas_kwh"].values
yP, pP = gas[P], z["gas_pP"]
yH, pc = z["gas_y"], z["gas_corrected"]


def imr(p, k=0.085):
    s = k * p
    a = (U - p) / s
    return p - s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def hats(x, knots):
    """hat basis on x with the given knots; first knot's hat is dropped so f = 0 at/below knots[0]; last is a ramp"""
    B = []
    for j in range(1, len(knots)):
        lo = knots[j - 1]
        hi = knots[j + 1] if j + 1 < len(knots) else None
        up = np.clip((x - lo) / (knots[j] - lo), 0, 1)
        if hi is None:
            b = np.where(x <= knots[j], up, 1.0)
        else:
            b = np.where(x <= knots[j], up, np.clip((hi - x) / (hi - knots[j]), 0, 1))
        B.append(b)
    return np.column_stack(B)


def nr(e):
    return np.sqrt(np.mean(e ** 2)) / SIG


base = nr(yH - pc)
KN = {"pw6": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1], "pw9": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1],
      "pw-low": [0.5, 0.6, 0.7, 0.8, 0.85]}
for name, kn in KN.items():
    for start in ("raw", "imr"):
        f0P = pP if start == "raw" else imr(pP)
        f0H = pc if start == "raw" else imr(pc)
        BP, BH = hats(pP / U, kn), hats(pc / U, kn)
        c = np.linalg.lstsq(BP, yP - f0P, rcond=None)[0]
        d_p = (nr(yH - (f0H + BH @ c)) - base) / 5
        outs = []
        for rep in range(10):
            pn = f0H.copy()
            for a, b in KFold(5, shuffle=True, random_state=rep).split(yH):
                cc = np.linalg.lstsq(BH[a], yH[a] - f0H[a], rcond=None)[0]
                pn[b] = f0H[b] + BH[b] @ cc
            outs.append((nr(yH - pn) - base) / 5)
        print(f"{name:7s} start={start}: P-fit coefs {np.round(c / 1e3, 1)}k -> H dTotal {d_p:+.8f} | H cross-fit "
              f"{np.mean(outs):+.8f} (sd {np.std(outs):.8f})")
print(f"IMR alone (k 0.085): H dTotal {(nr(yH - imr(pc)) - base) / 5:+.8f}")
