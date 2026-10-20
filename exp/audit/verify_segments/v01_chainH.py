"""Independent re-derivation of the gas-cap truncated-normal (IMR) calibration claim on the holdout chain.

Own implementation (scipy.special.log_ndtr, continuous k search), no code reused from exp/audit/segments/.
1. P labels recovered from chain_H itself (y_P = pP + rP), checked against the parquet column.
2. k fitted on the P-side teacher only (no H labels), scored on H corrected (honest), plus the teacher view.
3. 5-fold x 20 cross-fit of k on H corrected.
4. EUI through g = eui * A with the gas k.
5. Fragility: per-row SSE contributions, gain after dropping the best rows, sign counts.
6. Calibration check per bin: observed mean residual vs mean adjustment (P teacher, H teacher, H corrected, full VC OOF).
7. Gain curve over k on H (is the optimum flat / is a misestimated k still safe?).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/v01_chainH.py \
       > exp/audit/verify_segments/v01_chainH.log
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

SG, SE = 151750.04, 40.65
U = 1e6
LOG2PI = 0.5 * np.log(2 * np.pi)


def cal(g, k, cap=U):
    """truncated-normal mean below cap: g - s * phi(a)/Phi(a), s = k g, a = (cap - g)/s; only g > 0 touched"""
    g = np.asarray(g, dtype=float)
    out = g.copy()
    m = g > 0
    s = k * g[m]
    a = (cap - g[m]) / s
    lam = np.exp(-0.5 * a * a - LOG2PI - log_ndtr(a))
    out[m] = g[m] - s * lam
    return out


def fit_k(y, p, sel_lo=3e5):
    sel = p > sel_lo
    y, p = y[sel], p[sel]
    grid = np.arange(0.005, 0.3001, 0.0025)
    sse = np.array([np.sum((y - cal(p, k)) ** 2) for k in grid])
    k0 = grid[int(np.argmin(sse))]
    r = minimize_scalar(lambda k: np.sum((y - cal(p, k)) ** 2), bounds=(max(0.002, k0 - 0.005), k0 + 0.005),
                        method="bounded", options=dict(xatol=1e-5))
    return float(r.x)


def nr(e, sig):
    return float(np.sqrt(np.mean(e ** 2)) / sig)


z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH = A[H]
out = {}

# 1. labels
yP = z["gas_pP"] + z["gas_rP"]
gas_pq = pd.read_parquet("cache/train.parquet", columns=["target_macro_gas_kwh"])["target_macro_gas_kwh"].values
print(f"y_P from chain vs parquet: max |diff| {np.max(np.abs(yP - gas_pq[P])):.3e}; H y vs parquet max |diff| "
      f"{np.max(np.abs(z['gas_y'] - gas_pq[H])):.3e}; train gas max {gas_pq.max():.1f}, n>=1e6 {int((gas_pq >= 1e6).sum())}")
print(f"n P {len(P)} n H {len(H)} overlap {len(np.intersect1d(P, H))}")
del gas_pq

# 2. P-fit -> H
pP = z["gas_pP"]
kP = fit_k(yP, pP)
yH, pt, pc = z["gas_y"], z["gas_teacher"], z["gas_corrected"]
b_c, b_t = nr(yH - pc, SG), nr(yH - pt, SG)
d_c = (nr(yH - cal(pc, kP), SG) - b_c) / 5
d_t = (nr(yH - cal(pt, kP), SG) - b_t) / 5
print(f"\n[gas] k fitted on P teacher only: {kP:.5f}")
print(f"  H corrected base {b_c:.6f} -> dTotal {d_c:+.8f};  H teacher base {b_t:.6f} -> dTotal {d_t:+.8f}")
out["kP"] = kP
out["gas_dTotal_Pfit_Hcorr"] = d_c
out["gas_dTotal_Pfit_Hteach"] = d_t

# 3. cross-fit on H corrected
reps = []
ks = []
for rep in range(20):
    pn = pc.copy()
    for a, b in KFold(5, shuffle=True, random_state=100 + rep).split(yH):
        k = fit_k(yH[a], pc[a])
        ks.append(k)
        pn[b] = cal(pc[b], k)
    reps.append((nr(yH - pn, SG) - b_c) / 5)
reps = np.array(reps)
print(f"  H 5-fold x 20 cross-fit: dTotal mean {reps.mean():+.8f} sd {reps.std():.8f} worst {reps.max():+.8f}; "
      f"k range {min(ks):.4f}..{max(ks):.4f}")
out["gas_dTotal_Hcross"] = float(reps.mean())

# 4. EUI via g = eui * A, gas k
yE, pE, pEt = z["eui_y"], z["eui_corrected"], z["eui_teacher"]
bE = nr(yE - pE, SE)
gE = pE * AH
pEn = pE * cal(gE, kP) / np.where(gE > 0, gE, 1.0)
pEn = np.where(gE > 0, pEn, pE)
dE = (nr(yE - pEn, SE) - bE) / 5
print(f"\n[eui] via eui*A, k {kP:.5f}: H corrected base {bE:.6f} -> dTotal {dE:+.8f}")
# EUI own k fitted on P EUI teacher (gas scale), for reference
gEP = z["eui_pP"] * A[P]
yEP = (z["eui_pP"] + z["eui_rP"]) * A[P]
kE = fit_k(yEP, gEP)
pEn2 = np.where(gE > 0, pE * cal(gE, kE) / np.where(gE > 0, gE, 1.0), pE)
print(f"  own k fitted on P EUI teacher {kE:.5f} -> dTotal {(nr(yE - pEn2, SE) - bE) / 5:+.8f}")
out["eui_dTotal_Pfit_Hcorr"] = dE
out["combined_Pfit_Hcorr"] = d_c + dE
print(f"\n[combined, P-fit k] gas + eui dTotal on H corrected: {d_c + dE:+.8f}")

# 5. fragility
pn = cal(pc, kP)
dsse = (yH - pn) ** 2 - (yH - pc) ** 2
order = np.argsort(dsse)
tot = dsse.sum()
print(f"\n[fragility gas] rows with |change| > 1 kWh: {int(np.sum(np.abs(pn - pc) > 1))}; "
      f"helped {int(np.sum(dsse < -1e6))}, hurt {int(np.sum(dsse > 1e6))} (|dSSE| > 1e6)")
for j in range(8):
    i = order[j]
    print(f"   best #{j}: p {pc[i]:.0f} -> {pn[i]:.0f}, y {yH[i]:.0f}, share of total dSSE {dsse[i] / tot:.3f}")
for j in range(5):
    i = order[-1 - j]
    print(f"   worst #{j}: p {pc[i]:.0f} -> {pn[i]:.0f}, y {yH[i]:.0f}, dSSE/|total| {dsse[i] / abs(tot):+.3f}")
n = len(yH)
sse0 = np.sum((yH - pc) ** 2)
for drop in (0, 1, 3, 5, 10):
    keep = np.ones(n, bool)
    keep[order[:drop]] = False
    g = (np.sqrt(np.sum(((yH - pn) ** 2)[keep]) / keep.sum()) - np.sqrt(np.sum(((yH - pc) ** 2)[keep]) / keep.sum())) / SG / 5
    print(f"   drop the {drop:2d} most-helped rows: gas dTotal {g:+.8f}")
out["frag_top5_share"] = float(dsse[order[:5]].sum() / tot)

# 6. calibration check by bin
full = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
views = [("P teacher (P-OOF VC)", yP, pP), ("H teacher", yH, pt), ("H corrected", yH, pc),
         ("full-train VC OOF", full["gas_y"], full["gas_p"])]
bins = [0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 9]
kF = fit_k(full["gas_y"], full["gas_p"])
print(f"\n[calibration] k fitted per view: full VC OOF {kF:.5f} (P teacher {kP:.5f})")
for name, y, p in views:
    print(f"  {name}: bin of p/1e6 | n | mean resid y-p | mean adjustment p'-p (k {kP:.4f}) | t of resid")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p / U >= lo) & (p / U < hi)
        if m.sum() < 2:
            continue
        r = y[m] - p[m]
        adj = cal(p[m], kP) - p[m]
        t = r.mean() / (r.std(ddof=1) / np.sqrt(m.sum()))
        print(f"     [{lo:.2f},{hi:.2f}) n {int(m.sum()):4d}  resid {r.mean():+9.0f}  adj {adj.mean():+9.0f}  t {t:+5.2f}")
# full VC OOF: k fitted 5-fold cross on full OOF, gain
yF, pF = full["gas_y"], full["gas_p"]
bF = nr(yF - pF, SG)
rs = []
for rep in range(5):
    pn2 = pF.copy()
    for a, b in KFold(5, shuffle=True, random_state=200 + rep).split(yF):
        pn2[b] = cal(pF[b], fit_k(yF[a], pF[a]))
    rs.append((nr(yF - pn2, SG) - bF) / 5)
print(f"  full VC OOF (uncorrected VC blend, in-sample VC weights): base {bF:.6f}; cross-fitted k gas dTotal "
      f"{np.mean(rs):+.8f} (sd {np.std(rs):.8f}); with P-fit k {kP:.4f}: {(nr(yF - cal(pF, kP), SG) - bF) / 5:+.8f}")
yFE, pFE = full["eui_y"], full["eui_p"]
AF = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06"])["bld_dim_06"].values
gF = pFE * AF
pFEn = np.where(gF > 0, pFE * cal(gF, kP) / np.where(gF > 0, gF, 1.0), pFE)
print(f"  full VC OOF eui via eui*A, P-fit k: dTotal {(nr(yFE - pFEn, SE) - nr(yFE - pFE, SE)) / 5:+.8f}")
out["fullvc_gas_cross"] = float(np.mean(rs))

# 7. gain curve over k on H corrected (gas + eui)
print("\n[k curve on H corrected] k | gas dTotal | eui dTotal")
for k in (0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.2, 0.3):
    dg = (nr(yH - cal(pc, k), SG) - b_c) / 5
    de = (nr(yE - np.where(gE > 0, pE * cal(gE, k) / np.where(gE > 0, gE, 1.0), pE), SE) - bE) / 5
    print(f"   {k:.2f}  {dg:+.8f}  {de:+.8f}")

json.dump(out, open("exp/audit/verify_segments/v01_chainH.json", "w"), indent=1)
