"""verify2_integrity / step 7 (extra, beyond the claim): train elec per floor area is hard-bounded to [5, 150) kWh/m2
(0 rows outside, flat density up to 150). Is the honest chain elec prediction biased near these bounds, and does the
same truncated-normal shrink (two-sided, bounds 5*A and 150*A on the elec scale) gain anything honestly?
Fit on P (P-OOF teacher) -> score on H (corrected chain); plus 2-fold cross-fit inside H (50 seeds).
Also: does adding the EUI >= 5 lower bound change the claimed EUI shrink?
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/v7_elec_bound.py
"""
import json

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

SIG = {"elec": 41725.37, "eui": 40.65, "gas": 151750.04}
LO, HI = 5.0, 150.0
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
res = {}


def parts(metric):
    pP, yP = z[f"{metric}_pP"], z[f"{metric}_pP"] + z[f"{metric}_rP"]
    pH, yH = z[f"{metric}_corrected"], z[f"{metric}_y"]
    return (pP, yP, A[P], M[P]), (pH, yH, A[H], M[H])


# 1. bias by band of predicted elec / A
(pP, yP, AP, MP), (pH, yH, AH, MH) = parts("elec")
bands = [(0, 7), (7, 10), (10, 100), (100, 120), (120, 135), (135, 145), (145, 150), (150, 1e9)]
bias = {}
for name, p, y, a in (("H_corrected", pH, yH, AH), ("P_teacher", pP, yP, AP)):
    for lo, hi in bands:
        m = (p / a >= lo) & (p / a < hi)
        r = y[m] - p[m]
        bias[f"{name}_{lo}-{hi}"] = dict(n=int(m.sum()), mean_res=round(float(r.mean()), 1) if m.any() else None,
                                         se=round(float(r.std(ddof=1) / np.sqrt(m.sum())), 1) if m.sum() > 1 else None,
                                         mean_A=round(float(a[m].mean()), 1) if m.any() else None)
res["elec_bias_by_pred_per_A"] = bias
for k, v in bias.items():
    print(f"  {k:>26} {v}", flush=True)

# 2. variance model on P
m = (pP > 0) & (yP > 0)
lr2 = np.log(yP[m] / pP[m]) ** 2
a0, b0 = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), 1 / MP[m]]), lr2, rcond=None)[0]
res["elec_var_P"] = dict(a0=float(a0), b0=float(b0))
print("elec variance model P", res["elec_var_P"], flush=True)


def shift(p, Mv, a, k, lo=True, hi=True):
    """unit-c shift: mean of normal(p, s) truncated to [LO*A, HI*A) minus p"""
    s = np.maximum(k * np.abs(p) * np.sqrt(np.maximum(a0 + b0 / Mv, 1e-6)), 1.0)
    al = (LO * a - p) / s if lo else np.full(len(p), -np.inf)
    be = (HI * a - p) / s if hi else np.full(len(p), np.inf)
    # log(Phi(be) - Phi(al)) computed stably
    lb, la = log_ndtr(be), log_ndtr(al)
    logZ = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))
    num = np.exp(norm.logpdf(al) - logZ) - np.exp(norm.logpdf(be) - logZ)
    return s * num


KG = [0.3, 0.5, 0.7, 0.9, 1.1]


def fit(p, y, Mv, a, lo, hi):
    best = None
    for k in KG:
        t = shift(p, Mv, a, k, lo, hi)
        den = float(t @ t)
        c = float((y - p) @ t) / den if den > 0 else 0.0
        sse = float(((y - p - c * t) ** 2).sum())
        if best is None or sse < best[2]:
            best = (k, c, sse)
    return best[:2]


def nr(y, p):
    return np.sqrt(np.mean((y - p) ** 2)) / SIG["elec"]


base = nr(yH, pH)
out = {"base_H": float(base)}
for name, lo, hi in (("both", True, True), ("upper_only", False, True), ("lower_only", True, False)):
    k, c = fit(pP, yP, MP, AP, lo, hi)
    q = np.maximum(pH + c * shift(pH, MH, AH, k, lo, hi), 0)
    d1 = float((nr(yH, q) - base) / 5)
    cf = []
    for seed in range(50):
        f = np.random.default_rng(300 + seed).integers(0, 2, len(H))
        qq = np.empty(len(H))
        for j in range(2):
            tr_, te_ = f != j, f == j
            kk, cc = fit(pH[tr_], yH[tr_], MH[tr_], AH[tr_], lo, hi)
            qq[te_] = np.maximum(pH[te_] + cc * shift(pH[te_], MH[te_], AH[te_], kk, lo, hi), 0)
        cf.append(float((nr(yH, qq) - base) / 5))
    cf = np.array(cf)
    out[name] = dict(k_P=k, c_P=c, D1_fitP_scoreH_total=d1, H_2fold_mean=float(cf.mean()), H_2fold_max=float(cf.max()),
                     H_2fold_frac_neg=float((cf < 0).mean()))
    print(name, out[name], flush=True)
res["elec_shrink"] = out

# 4. EUI lower bound (EUI >= 5) on top of nothing (its own effect), fitted on P, scored on H
(pPe, yPe, _, MPe), (pHe, yHe, _, MHe) = parts("eui")
s_fn = lambda p, Mv, k: np.maximum(k * p * np.sqrt(np.maximum(0.008 + 0.39 / Mv, 1e-6)), 1e-3)  # noqa: E731
best = None
for k in KG:
    s = s_fn(pPe, MPe, k)
    al = (LO - pPe) / s
    t = s * np.exp(norm.logpdf(al) - log_ndtr(-al))
    c = float((yPe - pPe) @ t) / float(t @ t)
    sse = float(((yPe - pPe - c * t) ** 2).sum())
    if best is None or sse < best[2]:
        best = (k, c, sse)
k, c = best[:2]
s = s_fn(pHe, MHe, k)
al = (LO - pHe) / s
q = pHe + c * s * np.exp(norm.logpdf(al) - log_ndtr(-al))
res["eui_lower_bound"] = dict(k=k, c=c, n_pred_lt_10_H=int((pHe < 10).sum()),
                              change_total=float((np.sqrt(np.mean((yHe - q) ** 2)) -
                                                  np.sqrt(np.mean((yHe - pHe) ** 2))) / SIG["eui"] / 5))
print("eui lower bound", res["eui_lower_bound"], flush=True)
json.dump(res, open("exp/audit/verify2_integrity/v7_elec_bound.json", "w"), indent=1)
