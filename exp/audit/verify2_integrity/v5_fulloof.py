"""verify2_integrity / step 5: does the near-bound bias exist for the FULL-data members (the models behind the test
predictions), and does the claimed P-fitted shrink transfer to them?
Teacher = varying-coefficient NNLS blend (spec A2xM2xp2, as the pipeline) of the v19vce gas members' full-data OOF,
solved with an own low-memory Gram/Cholesky NNLS (same objective as blend.nnls_blend_vc). No residual correction.
Scores: H rows (labels never used by the claimed parameter fit), P rows, and a 5-fold cross-fit of (k, c) on all rows.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/v5_fulloof.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import nnls
from scipy.stats import norm

sys.path.insert(0, "final/src")
from blend import disagreement, vc_basis  # noqa: E402
from common import holdout_split  # noqa: E402

U = 1e6
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
CL = dict(a0=0.008064793880232542, b0=0.3929692979589114, k=0.7, c_gas=1.0126184473794142, c_eui=1.141696475118523)
KGRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12"])
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
yE, yG = tr[EUI].values, tr[GAS].values
n = len(tr)
Pz, Hz = holdout_split(n)
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
Po = np.column_stack([np.load(f"exp/oof/{m}.npy") for m in mem])
res = {"members": len(mem)}


def gram_nnls(X_fn, y, s, ncol, chunk=40000):
    G = np.zeros((ncol, ncol))
    b = np.zeros(ncol)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        Xc = X_fn(sl) * s[sl, None]
        G += Xc.T @ Xc
        b += Xc.T @ (y[sl] * s[sl])
    G += np.eye(ncol) * 1e-10 * np.trace(G) / ncol  # tiny ridge for numerical safety
    L = cholesky(G, lower=True)
    w, _ = nnls(L.T, solve_triangular(L, b, lower=True))
    return w


def vc_teacher(scale_A):
    s = A if scale_A else np.ones(n)
    w0 = gram_nnls(lambda sl: Po[sl], yE, s, Po.shape[1])
    p0 = Po @ w0
    v = dict(A=A, M=M, p=p0, d=disagreement(Po))
    B = vc_basis("A2xM2xp2", v, v)
    k = B.shape[1]
    W = gram_nnls(lambda sl: np.concatenate([Po[sl] * B[sl, [j]] for j in range(k)], axis=1), yE, s, k * Po.shape[1])
    W = W.reshape(k, Po.shape[1]).T
    return p0, ((Po @ W) * B).sum(axis=1)


g0, gvc = vc_teacher(True)
pred = {"gas": gvc * A}  # gas scale
e0, evc = vc_teacher(False)
pred["eui"] = evc * A  # EUI mapped to the gas scale
res["oof_nrmse"] = dict(gas_global=float(np.sqrt(np.mean((yG - g0 * A) ** 2)) / SIG["gas"]),
                        gas_vc=float(np.sqrt(np.mean((yG - pred["gas"]) ** 2)) / SIG["gas"]),
                        eui_global=float(np.sqrt(np.mean((yE - e0) ** 2)) / SIG["eui"]),
                        eui_vc=float(np.sqrt(np.mean((yE - evc) ** 2)) / SIG["eui"]))
print("oof nrmse", res["oof_nrmse"], flush=True)
del Po
np.savez_compressed("exp/audit/verify2_integrity/v5_fullvc_teacher.npz", gas=pred["gas"], eui=pred["eui"])

bands = [(7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 3e6)]
bias = {}
for metric in ("gas", "eui"):
    p = pred[metric]
    for part, idx in (("all", np.arange(n)), ("H", Hz)):
        for lo, hi in bands:
            m = idx[(p[idx] >= lo) & (p[idx] < hi)]
            r = yG[m] - p[m]
            bias[f"{metric}_{part}_{int(lo / 1e3)}k-{int(hi / 1e3)}k"] = dict(
                n=int(len(m)), mean_res=round(float(r.mean()), 1) if len(m) else None,
                se=round(float(r.std(ddof=1) / np.sqrt(len(m))), 1) if len(m) > 1 else None)
res["bias_by_band"] = bias
for k, v in bias.items():
    print(f"  {k:>28} {v}", flush=True)


def tfun(pg, Mv, a0, b0, k):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / Mv, 1e-6)), 1.0)
    a = (U - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def score(metric, q, idx):
    f = A[idx] if metric == "eui" else 1.0
    return np.sqrt(np.mean(((yG[idx] - q[idx]) / f) ** 2)) / SIG[metric]


# claimed P-fitted parameters applied to the full-data OOF teacher
cl = {}
for part, idx in (("H", Hz), ("P", Pz), ("all", np.arange(n))):
    tot = 0.0
    for metric in ("gas", "eui"):
        p = pred[metric]
        q = np.maximum(p + CL[f"c_{metric}"] * tfun(p, M, CL["a0"], CL["b0"], CL["k"]), 0)
        ch = (score(metric, q, idx) - score(metric, p, idx)) / 5
        cl[f"{part}_{metric}"] = float(ch)
        tot += ch
    cl[f"{part}_total"] = float(tot)
res["claimed_params_on_fulloof"] = cl
print("claimed params on full OOF", cl, flush=True)


def fit_kc(p, y, f, Mv):
    best = None
    for k in KGRID:
        t = tfun(p, Mv, CL["a0"], CL["b0"], k) / f
        r = (y - p) / f
        c = float(r @ t) / float(t @ t)
        sse = float(((r - c * t) ** 2).sum())
        if best is None or sse < best[2]:
            best = (k, c, sse)
    return best


# 5-fold cross-fit of (k, c) on the full OOF (10 seeds)
cf = []
for seed in range(10):
    fold = np.random.default_rng(777 + seed).integers(0, 5, n)
    tot, kcs = 0.0, []
    for metric in ("gas", "eui"):
        p = pred[metric]
        f = A if metric == "eui" else np.ones(n)
        q = np.empty(n)
        for j in range(5):
            a, b = fold != j, fold == j
            k, c, _ = fit_kc(p[a], yG[a], f[a], M[a])
            kcs.append((metric, k, round(c, 3)))
            q[b] = np.maximum(p[b] + c * tfun(p[b], M[b], CL["a0"], CL["b0"], k), 0)
        tot += (score(metric, q, np.arange(n)) - score(metric, p, np.arange(n))) / 5
    cf.append(dict(seed=seed, total=float(tot), kc=kcs[:2] + kcs[5:7]))
tots = np.array([c["total"] for c in cf])
res["crossfit_fulloof_5fold"] = dict(mean=float(tots.mean()), min=float(tots.min()), max=float(tots.max()),
                                     frac_negative=float((tots < 0).mean()), example_kc=cf[0]["kc"])
print("cross-fit full OOF", res["crossfit_fulloof_5fold"], flush=True)
# full-OOF optimum (in-sample, for reference only): k, c for each metric
res["fulloof_insample_kc"] = {m: fit_kc(pred[m], yG, A if m == "eui" else np.ones(n), M)[:2] for m in ("gas", "eui")}
print("in-sample k,c (reference only)", res["fulloof_insample_kc"], flush=True)
json.dump(res, open("exp/audit/verify2_integrity/v5_fulloof.json", "w"), indent=1)
print("wrote exp/audit/verify2_integrity/v5_fulloof.json")
