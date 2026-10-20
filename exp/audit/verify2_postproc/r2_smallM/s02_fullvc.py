"""Full-train VC blend (A2xM2xp2) of the v19vce members, rebuilt independently with a chunked TSQR-NNLS
(memory-light; same objective as blend.nnls_blend_vc), plus a fully nested version (VC blend fitted on 4/5 of the
train rows, applied to the remaining 1/5; fold seed 2026, not the production seed).

Checks: TSQR-NNLS vs production nnls_blend_vc on a 40k-row subsample.
Writes s02_fullvc.npz: pf_<m> (in-sample full fit on member OOF), pa_<m> (test), pn_<m> (nested), pnd_<m>
(nested fold-fit predictions are re-used later for deltas), fold.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/r2_smallM/s02_fullvc.py
"""
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

sys.path.insert(0, "final/src")
from blend import nnls_blend, nnls_blend_vc, predict_vc, vc_basis  # noqa: E402  (production code)

OUT = "exp/audit/verify2_postproc/r2_smallM"
SPEC = "A2xM2xp2"
GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {GAS: 151750.04, EUI: 40.65, EL: 41725.37}
t0 = time.time()

tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
cfg = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
Pm = {f: (np.column_stack([np.load(f"exp/oof/{r}.npy") for r in cfg[f]]).astype(np.float64),
          np.column_stack([np.load(f"exp/pred/{r}.npy") for r in cfg[f]]).astype(np.float64)) for f in ("gas", "elec")}
print("members gas", len(cfg["gas"]), "elec", len(cfg["elec"]), "shapes", Pm["gas"][0].shape, Pm["gas"][1].shape,
      flush=True)


def nnls_vc_tsqr(P, y, B, s=None, chunk=20000):
    """argmin_{W>=0} ||s * (y - sum_j B_j * (P @ W_j))|| via a streamed QR of [s*X | s*y] (X = members x basis)."""
    n, m = P.shape
    k = B.shape[1]
    s = np.ones(n) if s is None else s
    R = None
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        X = np.empty((b - a, m * k + 1))
        for j in range(k):
            X[:, j * m:(j + 1) * m] = P[a:b] * (B[a:b, j] * s[a:b])[:, None]
        X[:, -1] = y[a:b] * s[a:b]
        if R is not None:
            X = np.vstack([R, X])
        R = np.linalg.qr(X, mode="r")
    w, _ = nnls(R[:-1, :-1], R[:-1, -1], maxiter=50 * m * k)
    return w.reshape(k, m).T


def my_blend_vc(Pf, Pa, yf, vf0, va0, s):
    w = nnls_blend(Pf, yf, s)
    vf, va = dict(vf0, p=Pf @ w), dict(va0, p=Pa @ w)
    Bf, Ba = vc_basis(SPEC, vf, vf), vc_basis(SPEC, vf, va)
    W = nnls_vc_tsqr(Pf, yf, Bf, s)
    return predict_vc(Pf, W, Bf), predict_vc(Pa, W, Ba), W


# 0. check the TSQR solver against production nnls_blend_vc on a subsample (gas metric, row scale A)
rng = np.random.default_rng(5)
sub = np.sort(rng.choice(len(A), 40000, replace=False))
Pf0, yf0, s0 = Pm["gas"][0][sub], tr[EUI].values[sub], A[sub]
w0 = nnls_blend(Pf0, yf0, s0)
v0 = dict(A=A[sub], M=M[sub], p=Pf0 @ w0)
B0 = vc_basis(SPEC, v0, v0)
Wp = nnls_blend_vc(Pf0, yf0, B0, s0)
Wq = nnls_vc_tsqr(Pf0, yf0, B0, s0)
pp, pq = predict_vc(Pf0, Wp, B0) * s0, predict_vc(Pf0, Wq, B0) * s0
print(f"check TSQR vs production (40k rows, gas): max|dpred|/sigma {np.abs(pp - pq).max() / SIG[GAS]:.2e}, "
      f"rmse diff {np.sqrt(np.mean((yf0 * s0 - pp) ** 2)) - np.sqrt(np.mean((yf0 * s0 - pq) ** 2)):+.3e} kWh, "
      f"max|dW| {np.abs(Wp - Wq).max():.2e}", flush=True)
del Pf0, B0

out = {}
fold = np.random.default_rng(2026).integers(0, 5, len(A))
out["fold"] = fold
for metric, col, fam in (("gas", GAS, "gas"), ("eui", EUI, "gas"), ("elec", EL, "elec")):
    Pf, Pa = Pm[fam]
    yv = tr[EUI if fam == "gas" else EL].values
    s = A if metric == "gas" else None
    pf, pa, W = my_blend_vc(Pf, Pa, yv, dict(A=A, M=M), dict(A=At, M=Mt), s)
    if metric == "gas":
        pf, pa = pf * A, pa * At
    out[f"pf_{metric}"], out[f"pa_{metric}"] = pf, pa
    print(f"{metric}: full in-sample VC NRMSE {np.sqrt(np.mean((tr[col].values - pf) ** 2)) / SIG[col]:.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    # nested: fit on 4/5, apply to 1/5; keep the fit-side in-sample predictions too (for deltas)
    pn = np.zeros(len(A))
    for f in range(5):
        a, b = fold != f, fold == f
        sf = None if s is None else s[a]
        pfa, pfb, _ = my_blend_vc(Pf[a], Pf[b], yv[a], dict(A=A[a], M=M[a]), dict(A=A[b], M=M[b]), sf)
        if metric == "gas":
            pfa, pfb = pfa * A[a], pfb * A[b]
        pn[b] = pfb
        out[f"pfit_{metric}_{f}"] = pfa  # in-sample fit on the 4/5 rows (order = rows with fold != f)
    out[f"pn_{metric}"] = pn
    print(f"{metric}: nested VC NRMSE {np.sqrt(np.mean((tr[col].values - pn) ** 2)) / SIG[col]:.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)
np.savez(f"{OUT}/s02_fullvc.npz", **out)
print("wrote", f"{OUT}/s02_fullvc.npz", f"({time.time() - t0:.0f}s)")
