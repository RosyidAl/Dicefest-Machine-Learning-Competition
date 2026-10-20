"""Truncation-aware calibration near the gas < 1e6 cap (and elec / A < 150), honest evaluation.

Data facts (train targets): gas max 999,765 with flat density up to the edge and none above 1e6 (hard filter);
elec / A max 149.99, none above 150. Predictions near the cap are biased upward (truth is cut at the cap).

Calibrations (U = cap; for elec U_i = 150 A_i):
  clip   : p' = min(p, U)
  imr    : truncated-normal mean, p' = p - s phi(a) / Phi(a), s = k p, a = (U - p) / s   (one parameter k)
  imrc   : imr followed by clip at U
  ramp   : p' = p - b max(0, p - c U)                                                  (two parameters b, c)
Fitting: (i) on P-side teacher (OOF VC blend, no correction) -> scored on H corrected (fully honest, out-of-sample);
         (ii) cross-fitted on H corrected, 5-fold x 20 reps.
EUI: the gas-scale adjustment of eui * A applied as eui' = eui * (g'/g), with g = eui * A (same truncation, since
eui = gas / A exactly).  Effects as change of the TOTAL score (per-target NRMSE change / 5).

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_cap.py > exp/audit/segments/seg_cap.log
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
COL = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
ytr = pd.read_parquet("cache/train.parquet", columns=list(COL.values()))
KGRID = np.round(np.arange(0.01, 0.301, 0.005), 4)
BGRID = np.round(np.arange(0.0, 1.01, 0.05), 3)
CGRID = np.round(np.arange(0.80, 1.001, 0.01), 3)


def imr_adj(p, U, k):
    s = np.maximum(k * p, 1e-9)
    a = (U - p) / s
    # inverse Mills ratio phi(a)/Phi(a), stable for very negative a
    lam = np.exp(norm.logpdf(a) - norm.logcdf(a))
    return p - s * lam


def calib(kind, p, U, par):
    if kind == "clip":
        return np.minimum(p, U)
    if kind == "imr":
        return imr_adj(p, U, par[0])
    if kind == "imrc":
        return np.minimum(imr_adj(p, U, par[0]), U)
    if kind == "ramp":
        b, c = par
        return p - b * np.maximum(0.0, p - c * U)
    raise ValueError(kind)


def grid(kind):
    if kind == "clip":
        return [()]
    if kind in ("imr", "imrc"):
        return [(k,) for k in KGRID]
    return [(b, c) for b in BGRID for c in CGRID]


def fit(kind, y, p, U):
    best, bp = np.inf, None
    for par in grid(kind):
        sse = np.sum((y - calib(kind, p, U, par)) ** 2)
        if sse < best:
            best, bp = sse, par
    return bp


def nr(e, m):
    return np.sqrt(np.mean(e ** 2)) / SIG[m]


AH = A[H]
for m in ("gas", "elec"):
    yP = ytr[COL[m]].values[P]
    pP = z[f"{m}_pP"]
    yH, pc, pt = z[f"{m}_y"], z[f"{m}_corrected"], z[f"{m}_teacher"]
    UP = 1e6 if m == "gas" else 150.0 * A[P]
    UH = 1e6 if m == "gas" else 150.0 * AH
    base = nr(yH - pc, m)
    print(f"\n=== {m}: H corrected NRMSE {base:.6f}")
    for kind in ("clip", "imr", "imrc", "ramp"):
        parP = fit(kind, yP, pP, UP)
        pn = calib(kind, pc, UH, parP)
        d_hon = (nr(yH - pn, m) - base) / 5
        # also: fitted on P teacher, applied to H teacher (same-stage check)
        pnt = calib(kind, pt, UH, parP)
        d_t = (nr(yH - pnt, m) - nr(yH - pt, m)) / 5
        # cross-fitted on H corrected
        outs, pars = [], []
        for rep in range(20):
            en = yH - pc
            for a, b in KFold(5, shuffle=True, random_state=rep).split(yH):
                Ua = UH if np.isscalar(UH) else UH[a]
                Ub = UH if np.isscalar(UH) else UH[b]
                par = fit(kind, yH[a], pc[a], Ua)
                pars.append(par)
                en = en.copy()
                en[b] = yH[b] - calib(kind, pc[b], Ub, par)
            outs.append((nr(en, m) - base) / 5)
        parH = fit(kind, yH, pc, UH)
        print(f"  {kind:5s} fit on P teacher {parP} -> H corrected dTotal {d_hon:+.8f} (H teacher {d_t:+.8f}) | "
              f"cross-fit on H dTotal {np.mean(outs):+.8f} (sd {np.std(outs):.8f}, worst rep {np.max(outs):+.8f})  "
              f"all-H fit {parH}  rows changed {int(np.sum(np.abs(pn - pc) > 1e-6 * np.abs(pc)))}")
        if m == "gas":
            # EUI column: same gas-scale adjustment on g = eui * A
            yE, pE = z["eui_y"], z["eui_corrected"]
            g = pE * AH
            gn = calib(kind, g, UH, parP)
            pEn = pE * gn / g
            print(f"        EUI via eui*A, P-teacher params -> dTotal {(nr(yE - pEn, 'eui') - nr(yE - pE, 'eui')) / 5:+.8f}")
            if kind in ("imr", "imrc"):
                # bootstrap the honest (P-fit) H gain for gas
                rng = np.random.default_rng(0)
                e0, e1 = (yH - pc) ** 2, (yH - pn) ** 2
                bs = []
                for _ in range(2000):
                    i = rng.integers(0, len(yH), len(yH))
                    bs.append((np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())) / SIG[m] / 5)
                print(f"        bootstrap of the P-fit H gain: mean {np.mean(bs):+.8f} sd {np.std(bs):.8f}  "
                      f"P(gain>=0) {np.mean(np.array(bs) >= 0):.4f}")
