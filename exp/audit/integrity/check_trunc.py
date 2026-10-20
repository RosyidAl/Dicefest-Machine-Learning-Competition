"""Integrity audit 7: truncation of gas at T = 1e6 kWh (no train row >= 1e6; max 999,765).

If y = mu + e, e ~ N(0, s^2), observed only when y < T, the MSE-optimal prediction is E[y | y < T] = mu - s*lam(a),
lam(a) = phi(a)/Phi(a), a = (T - mu)/s. Our blend overpredicts near T (H mean residual -37k for p > 900k).
Variance model s = p * sqrt(a0 + b0/M) is fitted on P-side residuals (teacher, P-OOF), and a scale c on the
adjustment is fitted on P (teacher) -> applied to the honest corrected chain prediction on H. Never fitted on H,
except the explicit 5-fold cross-fitted variant reported for comparison.
EUI uses the same adjustment on the gas scale (EUI*A), then divided back by A.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_trunc.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
out = {}


def lam(a):
    return np.exp(norm.logpdf(a) - norm.logcdf(a))


def adj(p, s):
    """E[y | y < T] - p for y ~ N(p, s^2)"""
    s = np.maximum(s, 1.0)
    return -s * lam((T - p) / s)


# 1. variance model on P (teacher P-OOF residuals, gas scale)
pP, rP = z["gas_pP"], z["gas_rP"]
okP = pP > 1000
X = np.column_stack([np.ones(okP.sum()), 1.0 / M[P][okP]])
coef = np.linalg.lstsq(X, (rP[okP] / pP[okP]) ** 2, rcond=None)[0]
out["var_model_P"] = dict(a0=float(coef[0]), b0=float(coef[1]))


def sd(p, m):
    return np.abs(p) * np.sqrt(np.maximum(coef[0] + coef[1] / m, 1e-6))


# 2. P-side bias near T (teacher) and fitted scale c on P
aP = adj(pP, sd(pP, M[P]))
for lo in (8e5, 9e5, 9.5e5):
    msk = pP > lo
    out[f"P_teacher_pred_gt_{lo:.0f}"] = dict(n=int(msk.sum()), mean_resid=float(rP[msk].mean()),
                                               mean_adj=float(aP[msk].mean()))
cP = float(np.sum(aP * rP) / np.sum(aP * aP))
out["c_fitted_on_P"] = cP


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


# 3. apply to H (honest corrected chain prediction)
res = {}
for m in ("gas", "eui"):
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    scale = A[H] if m == "eui" else 1.0
    pg = p * scale  # gas scale
    a = adj(pg, sd(pg, M[H]))
    r = {"base": nr(y, p, m)}
    for name, c in (("c=1", 1.0), ("c=P", cP)):
        pa = (pg + c * a) / scale
        r[name] = dict(nrmse=nr(y, pa, m), delta_total=(nr(y, pa, m) - r["base"]) / 5,
                       rows_adj_gt_1pct=int((np.abs(c * a) > 0.01 * pg).sum()))
    # hard clip only
    pc = np.minimum(pg, T) / scale
    r["clip_only"] = dict(nrmse=nr(y, pc, m), delta_total=(nr(y, pc, m) - r["base"]) / 5)
    # 5-fold cross-fitted c on H (comparison only)
    pa = np.empty_like(p)
    cs = []
    for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(p):
        rr = (y * scale - pg)[tr_i]
        c = float(np.sum(a[tr_i] * rr) / np.sum(a[tr_i] ** 2))
        cs.append(c)
        pa[te_i] = (pg[te_i] + c * a[te_i]) / scale[te_i] if m == "eui" else pg[te_i] + c * a[te_i]
    r["c_crossfit_H"] = dict(cs=cs, nrmse=nr(y, pa, m), delta_total=(nr(y, pa, m) - r["base"]) / 5)
    # split-half stability for c=P
    pa = (pg + cP * a) / scale
    halves = []
    for h in range(2):
        idx = np.arange(len(p)) % 2 == h
        halves.append((nr(y[idx], pa[idx], m) - nr(y[idx], p[idx], m)) / 5)
    r["c=P_half_deltas"] = halves
    # bootstrap sd of delta for c=P
    rng = np.random.default_rng(0)
    e0, e1 = (y - p) ** 2, (y - pa) ** 2
    bs = []
    for _ in range(200):
        b = rng.integers(0, len(p), len(p))
        bs.append((np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] / 5)
    r["c=P_boot_sd"] = float(np.std(bs))
    r["c=P_boot_p_worse"] = float(np.mean(np.array(bs) > 0))
    res[m] = r
out["H"] = res

# 4. test side: how many submitted predictions move, and by how much
te = pd.read_parquet("data/cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
for m, col in (("gas", "target_macro_gas_kwh"), ("eui", "target_eui_gas_kwh_per_sqm")):
    p = sub[col].values
    pg = p * (At if m == "eui" else 1.0)
    a = cP * adj(pg, sd(pg, Mt))
    out[f"test_{m}"] = dict(rows_adj_gt_1pct=int((np.abs(a) > 0.01 * pg).sum()),
                            rows_pred_gt_900k=int((pg > 9e5).sum()),
                            rms_adj_over_sigma=float(np.sqrt(np.mean((a / (At if m == "eui" else 1.0)) ** 2)) / SIG[m]))
out["H_rows_pred_gt_900k"] = int((z["gas_corrected"] > 9e5).sum())
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_trunc.json", "w"), indent=1)
