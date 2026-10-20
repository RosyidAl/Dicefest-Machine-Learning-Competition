"""Verifier round 3 (leakage/optimism lens), step A: independent re-derivation of the integrity truncation-shrink
claim from exp/audit/chain_H.npz only (own code, no parameter read from the auditor's json files).

Claim: gas/EUI p -> p + c * adj_k(p), adj_k = -s phi(a)/Phi(a), a = (1e6 - p)/s, s = k p sqrt(a0 + b0/M);
a0, b0 (variance model), k (grid) and c (least squares) fitted on P (teacher P-OOF residuals) -> scored on the honest
corrected chain prediction on H; claimed total -2.556e-5.

Checks:
 0. P/H disjoint and covering train; P labels below 1e6; no H array used before scoring.
 1. own fit of a0, b0, k, c on P; H deltas per metric and total; paired bootstrap on H.
 2. calibration on H: gamma = sum(d e)/sum(d^2) (1 = the P-fitted bias model is exactly right on H); the empirical
    dSSE vs the model-expected -sum(d^2).
 3. leave-band-out: H gain from rows by prediction band (where does the gain come from).
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/r3_a_rederive.py
"""
import json

import numpy as np
from scipy.stats import norm

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
KGRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
OUT = "exp/audit/verify_integrity/r3_a_rederive.json"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
out = {}

# 0. structure
out["n_P"], out["n_H"] = int(len(P)), int(len(H))
out["P_H_disjoint"] = bool(len(np.intersect1d(P, H)) == 0)
out["P_H_cover_train"] = bool(len(np.union1d(P, H)) == len(A))
yPg = z["gas_pP"] + z["gas_rP"]
yPe = z["eui_pP"] + z["eui_rP"]
out["P_gas_label_max"] = float(yPg.max())
out["P_gas_vs_euiA_label_maxrel"] = float(np.max(np.abs(yPg - yPe * A[P]) / np.maximum(yPg, 1)))


# 1. fit on P only
def var_fit(p, r, m, filt=True):
    ok = p > 1000 if filt else np.ones(len(p), bool)
    X = np.column_stack([np.ones(ok.sum()), 1.0 / m[ok]])
    return np.linalg.lstsq(X, (r[ok] / p[ok]) ** 2, rcond=None)[0]


a0, b0 = var_fit(z["gas_pP"], z["gas_rP"], M[P])
out["var_model_P"] = dict(a0=float(a0), b0=float(b0))


def adj(pg, m, k):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / m, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


fit = {}
for m in ("gas", "eui"):
    scP = A[P] if m == "eui" else np.ones(len(P))
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    best = None
    grid = {}
    for k in KGRID:
        aP = adj(pP * scP, M[P], k) / scP
        c = float(aP @ rP / (aP @ aP))
        dsse = float(np.sum((rP - c * aP) ** 2) - np.sum(rP ** 2))
        grid[str(k)] = dict(c=c, dSSE_P=dsse)
        if best is None or dsse < best[2]:
            best = (k, c, dsse)
    fit[m] = dict(k=best[0], c=best[1], grid=grid)
out["fit_P"] = fit

# score on H (first use of H labels)
rng = np.random.default_rng(123)
B = [rng.integers(0, len(H), len(H)) for _ in range(1000)]
tot_bs = np.zeros(len(B))
res = {}
D = {}
for m in ("gas", "eui"):
    scH = A[H] if m == "eui" else np.ones(len(H))
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    k, c = fit[m]["k"], fit[m]["c"]
    d = c * adj(p * scH, M[H], k) / scH
    pa = p + d
    e = y - p
    D[m] = (d, e, p, y)
    base, new = nr(y, p, m), nr(y, pa, m)
    e0, e1 = e ** 2, (y - pa) ** 2
    bs = np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] / 5 for b in B])
    tot_bs += bs
    gamma = float(d @ e / (d @ d))
    dsse_emp = float(np.sum(e1 - e0))
    dsse_model = float(-np.sum(d ** 2))
    bands = {}
    for lo, hi in ((0, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 1e7)):
        g = (p * scH >= lo) & (p * scH < hi)
        bands[f"{lo:.0f}-{hi:.0f}"] = dict(n=int(g.sum()), dSSE_share=float(np.sum((e1 - e0)[g]) / dsse_emp),
                                           mean_resid=float(e[g].mean()) if g.any() else None,
                                           mean_d=float(d[g].mean()) if g.any() else None)
    res[m] = dict(k=k, c=c, base=base, new=new, delta_total=(new - base) / 5, boot_sd=float(bs.std()),
                  boot_p_worse=float((bs > 0).mean()), gamma_H=gamma, dSSE_emp=dsse_emp, dSSE_model=dsse_model,
                  rows_moved_gt_1pct=int((np.abs(d) > 0.01 * p).sum()), bands=bands)
res["total"] = dict(delta_total=res["gas"]["delta_total"] + res["eui"]["delta_total"], boot_sd=float(tot_bs.std()),
                    boot_p_worse=float((tot_bs > 0).mean()), boot_q05=float(np.quantile(tot_bs, 0.05)),
                    boot_q95=float(np.quantile(tot_bs, 0.95)))
out["H"] = res
out["claimed"] = dict(a0=0.008064793880232542, b0=0.3929692979589114, k=0.7, c_gas=1.0126184473794142,
                      c_eui=1.141696475118523, total=-2.56e-5)
out["matches_claim"] = bool(abs(res["total"]["delta_total"] - (-2.556206168642383e-05)) < 2e-7
                            and fit["gas"]["k"] == 0.7 and fit["eui"]["k"] == 0.7)
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
