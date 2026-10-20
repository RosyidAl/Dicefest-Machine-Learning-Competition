"""verify2_integrity / step 3: honest re-measurement of a truncation-aware shrink for gas and EUI (own code).
Shrink: p -> p + c * t(p), t(p) = -s * phi(a) / Phi(a), a = (U - p) / s, s = k * p * sqrt(a0 + b0 / M), U = 1e6
(mean of a normal(p, s) truncated above at U). EUI is shrunk on the gas scale (EUI * A) and divided back by A.
Designs (no design ever scores rows whose labels were used for the fit):
 D1  variance model, k and c fitted on P (P-OOF teacher)            -> scored on H (corrected chain prediction)
 D2  variance model on P; k, c fitted on one half of H               -> scored on the other half (2-fold, 50 seeds)
 D3  5-fold cross-fit of k, c on H (20 seeds)
 D4  fit-free: c = 1, variance model from P, k grid                   -> H
 D5  hard clip at U (no fit), and clip at U - d with d fitted on P     -> H
 D6  bootstrap of the D1 change on H (2000 resamples, paired)
All changes are in TOTAL-score units (per-target NRMSE change / 5).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/v3_shrink.py
"""
import json

import numpy as np
from scipy.stats import norm

U = 1e6
SIG = {"gas": 151750.04, "eui": 40.65}
KGRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
res = {}


def tfun(pg, Mv, a0, b0, k):
    """unit-c shift on the gas scale (<= 0)"""
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / Mv, 1e-6)), 1.0)
    a = (U - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def side(metric, part):
    """gas-scale prediction, gas-scale truth, metric scale factor (pred_metric = pg / f) and M of a part"""
    if part == "P":
        idx = P
        p, y = z[f"{metric}_pP"], z[f"{metric}_pP"] + z[f"{metric}_rP"]
    else:
        idx = H
        p, y = z[f"{metric}_corrected"], z[f"{metric}_y"]
    f = A[idx] if metric == "eui" else np.ones(len(idx))
    return p * f, y * f, f, M[idx]


def nrmse(y, p, metric):
    return np.sqrt(np.mean((y - p) ** 2)) / SIG[metric]


# ---- variance model on P only: E[log(y/p)^2] = a0 + b0 / M, fitted by OLS on rows far from the bound
pg, yg, _, Mp = side("gas", "P")
m = (pg < 6e5) & (pg > 0) & (yg > 0)
lr2 = np.log(yg[m] / pg[m]) ** 2
X = np.column_stack([np.ones(m.sum()), 1 / Mp[m]])
a0, b0 = np.linalg.lstsq(X, lr2, rcond=None)[0]
# robust alternative: winsorize the squared log residual at its 99.9th percentile
lr2w = np.minimum(lr2, np.quantile(lr2, 0.999))
a0w, b0w = np.linalg.lstsq(X, lr2w, rcond=None)[0]
res["variance_model_P"] = dict(a0=float(a0), b0=float(b0), a0_winsor=float(a0w), b0_winsor=float(b0w),
                               claimed=dict(a0=0.008064793880232542, b0=0.3929692979589114))
print("variance model on P:", res["variance_model_P"], flush=True)


def fit_kc(p, y, f, Mv, kgrid=KGRID, var=(a0, b0)):
    """for each k: least-squares c on the METRIC scale (weights 1/f^2 for EUI); pick k with the lowest SSE"""
    best = None
    for k in kgrid:
        t = tfun(p, Mv, var[0], var[1], k) / f
        r = (y - p) / f
        den = float(t @ t)
        c = float(r @ t) / den if den > 0 else 0.0
        sse = float(((r - c * t) ** 2).sum())
        if best is None or sse < best[2]:
            best = (k, c, sse)
    return best[0], best[1]


def apply(p, f, Mv, k, c, var=(a0, b0)):
    return np.maximum(p + c * tfun(p, Mv, var[0], var[1], k), 0) / f


base = {}
for metric in ("gas", "eui"):
    p, y, f, Mv = side(metric, "H")
    base[metric] = float(nrmse(y / f, p / f, metric))
res["base_H"] = base
print("base H", base, flush=True)

# ---- D1: everything fitted on P
d1 = {}
for metric in ("gas", "eui"):
    k, c = fit_kc(*side(metric, "P"))
    p, y, f, Mv = side(metric, "H")
    new = nrmse(y / f, apply(p, f, Mv, k, c), metric)
    d1[metric] = dict(k=k, c=c, nrmse=float(new), change_total=float((new - base[metric]) / 5))
    # also with the claimed parameters (k=0.7, their c) on the claimed variance model, as a cross-check
d1["total"] = d1["gas"]["change_total"] + d1["eui"]["change_total"]
res["D1_fit_P_score_H"] = d1
print("D1", d1, flush=True)

# claimed parameter set, re-implemented here (no reuse of their code)
cl = dict(a0=0.008064793880232542, b0=0.3929692979589114, k=0.7, c_gas=1.0126184473794142, c_eui=1.141696475118523)
d1c = {}
for metric in ("gas", "eui"):
    p, y, f, Mv = side(metric, "H")
    new = nrmse(y / f, apply(p, f, Mv, cl["k"], cl[f"c_{metric}"], (cl["a0"], cl["b0"])), metric)
    d1c[metric] = dict(nrmse=float(new), change_total=float((new - base[metric]) / 5))
d1c["total"] = d1c["gas"]["change_total"] + d1c["eui"]["change_total"]
res["D1_claimed_params_score_H"] = d1c
print("D1 claimed params", d1c, flush=True)

# ---- D2 / D3: cross-fitting inside H
def crossfit(nfold, seeds):
    out = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        fold = rng.integers(0, nfold, len(H))
        tot = 0.0
        per = {}
        for metric in ("gas", "eui"):
            p, y, f, Mv = side(metric, "H")
            q = np.empty(len(H))
            for j in range(nfold):
                a, b = fold != j, fold == j
                k, c = fit_kc(p[a], y[a], f[a], Mv[a])
                q[b] = apply(p[b], f[b], Mv[b], k, c)
            ch = float((nrmse(y / f, q, metric) - base[metric]) / 5)
            per[metric] = ch
            tot += ch
        out.append(dict(seed=seed, total=tot, **per))
    tots = np.array([o["total"] for o in out])
    return dict(mean_total=float(tots.mean()), sd_over_seeds=float(tots.std()), min=float(tots.min()),
                max=float(tots.max()), frac_negative=float((tots < 0).mean()),
                mean_gas=float(np.mean([o["gas"] for o in out])), mean_eui=float(np.mean([o["eui"] for o in out])))


res["D2_H_2fold_50seeds"] = crossfit(2, range(50))
print("D2", res["D2_H_2fold_50seeds"], flush=True)
res["D3_H_5fold_20seeds"] = crossfit(5, range(20))
print("D3", res["D3_H_5fold_20seeds"], flush=True)

# per-k honesty check: fixed k, c fitted on one half and scored on the other (both directions), 50 seeds
perk = {}
for k in KGRID:
    vals = []
    for seed in range(50):
        fold = np.random.default_rng(5000 + seed).integers(0, 2, len(H))
        tot = 0.0
        for metric in ("gas", "eui"):
            p, y, f, Mv = side(metric, "H")
            q = np.empty(len(H))
            for j in range(2):
                a, b = fold != j, fold == j
                _, c = fit_kc(p[a], y[a], f[a], Mv[a], kgrid=[k])
                q[b] = apply(p[b], f[b], Mv[b], k, c)
            tot += (nrmse(y / f, q, metric) - base[metric]) / 5
        vals.append(tot)
    vals = np.array(vals)
    perk[str(k)] = dict(mean=float(vals.mean()), max=float(vals.max()), frac_negative=float((vals < 0).mean()))
res["D2_per_k_2fold"] = perk
print("D2 per k", perk, flush=True)

# ---- D4: fit-free c = 1
d4 = {}
for k in KGRID:
    tot = 0.0
    for metric in ("gas", "eui"):
        p, y, f, Mv = side(metric, "H")
        tot += (nrmse(y / f, apply(p, f, Mv, k, 1.0), metric) - base[metric]) / 5
    d4[str(k)] = float(tot)
res["D4_c1_H"] = d4
print("D4", d4, flush=True)

# ---- D5: hard clips
d5 = {}
for metric in ("gas", "eui"):
    p, y, f, Mv = side(metric, "H")
    d5[f"{metric}_clip_U"] = float((nrmse(y / f, np.minimum(p, U) / f, metric) - base[metric]) / 5)
    pp, yp, fp, _ = side(metric, "P")
    caps = np.arange(8.5e5, 1.0001e6, 5e3)
    sse = [float((((yp - np.minimum(pp, cap)) / fp) ** 2).sum()) for cap in caps]
    cap = float(caps[int(np.argmin(sse))])
    d5[f"{metric}_cap_fitP"] = cap
    d5[f"{metric}_clip_capP"] = float((nrmse(y / f, np.minimum(p, cap) / f, metric) - base[metric]) / 5)
d5["total_clip_U"] = d5["gas_clip_U"] + d5["eui_clip_U"]
d5["total_clip_capP"] = d5["gas_clip_capP"] + d5["eui_clip_capP"]
res["D5_clips"] = d5
print("D5", d5, flush=True)

# ---- D6: paired bootstrap of the D1 change on H
rng = np.random.default_rng(42)
sq = {}
for metric in ("gas", "eui"):
    p, y, f, Mv = side(metric, "H")
    q = apply(p, f, Mv, d1[metric]["k"], d1[metric]["c"])
    sq[metric] = (((y - p) / f) ** 2, (y / f - q) ** 2)
B = 2000
bt = np.zeros(B)
n = len(H)
for b in range(B):
    i = rng.integers(0, n, n)
    for metric in ("gas", "eui"):
        s0, s1 = sq[metric][0][i].mean(), sq[metric][1][i].mean()
        bt[b] += (np.sqrt(s1) - np.sqrt(s0)) / SIG[metric] / 5
res["D6_bootstrap_D1"] = dict(mean=float(bt.mean()), sd=float(bt.std()), p_worse=float((bt >= 0).mean()))
print("D6", res["D6_bootstrap_D1"], flush=True)

# where does the D1 gain come from (gas): rows with |shift| > 1% of p
p, y, f, Mv = side("gas", "H")
q = apply(p, f, Mv, d1["gas"]["k"], d1["gas"]["c"])
mv = np.abs(q - p) > 0.01 * p
res["D1_gas_moved_rows_H"] = dict(n=int(mv.sum()), sse_before=float(((y - p)[mv] ** 2).sum()),
                                  sse_after=float(((y - q)[mv] ** 2).sum()),
                                  n_improved=int((np.abs(y - q) < np.abs(y - p))[mv].sum()))
print("D1 moved rows", res["D1_gas_moved_rows_H"], flush=True)
json.dump(res, open("exp/audit/verify2_integrity/v3_shrink.json", "w"), indent=1)
print("wrote exp/audit/verify2_integrity/v3_shrink.json")
