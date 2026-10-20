"""Verifier (leakage/optimism lens) 3: expected TEST gain of the truncation shrink, reweighting per-band gains to the
test's distribution of predictions near the bound (the submitted test predictions are less extreme near 1e6 than the
holdout-chain predictions: rate >= 950k 1.63 vs 2.72 per 10k rows).

Per-row squared-error change de = (y - p_new)^2 - (y - p)^2 with the claimed parameters, averaged per band of the
gas-scale prediction (gas: p; EUI: p*A), from three label sources:
  chain_H     : honest chain prediction on H (the claim's own setting)
  fullOOF_H   : v19vce VC blend of full-data OOF members (90%-data models), H rows only (no H label used for params)
  fullOOF_all : same, all rows (P rows share labels with the P-teacher fit -> mildly optimistic)
then expected test delta_total = sum_b share_b(test) * mean_b(de) / (2 * RMSE * sigma) / 5 (linearized, RMSE of the
source). Bootstrap over source rows for the sd.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/v3_reweight.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.stats import norm

sys.path.insert(0, "final/src")
from blend import vc_basis  # noqa: E402

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
CL = dict(a0=0.008064793880232542, b0=0.3929692979589114, k=0.7, c_gas=1.0126184473794142, c_eui=1.141696475118523)
EDGES = np.array([0, 7e5, 8e5, 8.5e5, 9e5, 9.5e5, 1e6, np.inf])
OUT = "exp/audit/verify_integrity/v3_reweight.json"


def adj(pg, m):
    s = np.maximum(CL["k"] * np.abs(pg) * np.sqrt(np.maximum(CL["a0"] + CL["b0"] / m, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def vc_lowmem(Pf, Pa, yf, spec, vf, va, s=None):
    s = np.ones(len(yf)) if s is None else s
    w = nnls(Pf * s[:, None], yf * s, maxiter=5000)[0]
    vf = dict(vf, p=Pf @ w)
    va = dict(va, p=Pa @ w)
    Bf, Ba = vc_basis(spec, vf, vf), vc_basis(spec, vf, va)
    nb, nm = Bf.shape[1], Pf.shape[1]
    G = np.zeros((nb * nm, nb * nm))
    g = np.zeros(nb * nm)
    for j in range(nb):
        Xj = Pf * (Bf[:, j] * s)[:, None]
        g[j * nm:(j + 1) * nm] = Xj.T @ (yf * s)
        for l in range(j, nb):
            blk = Xj.T @ (Pf * (Bf[:, l] * s)[:, None])
            G[j * nm:(j + 1) * nm, l * nm:(l + 1) * nm] = blk
            G[l * nm:(l + 1) * nm, j * nm:(j + 1) * nm] = blk.T
    G += np.eye(len(G)) * 1e-12 * np.trace(G) / len(G)
    L = np.linalg.cholesky(G)
    W = nnls(L.T, np.linalg.solve(L, g), maxiter=20000)[0].reshape(nb, nm).T
    return ((Pf @ W) * Bf).sum(1), ((Pa @ W) * Ba).sum(1)


tr = pd.read_parquet("data/cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EUI])
te = pd.read_parquet("data/cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
yg, ye = tr[GAS].values, tr[EUI].values
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H = z["P"], z["H"]
names = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
Pf = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in names])
Pa = np.column_stack([np.load(f"exp/pred/{r}.npy") for r in names])
fg, _ = vc_lowmem(Pf, Pa, ye, "A2xM2xp2", dict(A=A, M=M), dict(A=At, M=Mt), A)
fe, _ = vc_lowmem(Pf, Pa, ye, "A2xM2xp2", dict(A=A, M=M), dict(A=At, M=Mt), None)
del Pf, Pa
fg, fe = np.clip(fg * A, 0, None), np.clip(fe, 0, None)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
test_pg = {"gas": sub[GAS].values, "eui": sub[EUI].values * At}

src = {
    "chain_H": dict(gas=(z["gas_y"], np.clip(z["gas_corrected"], 0, None), M[H], np.ones(len(H))),
                    eui=(z["eui_y"], np.clip(z["eui_corrected"], 0, None), M[H], A[H])),
    "fullOOF_H": dict(gas=(yg[H], fg[H], M[H], np.ones(len(H))), eui=(ye[H], fe[H], M[H], A[H])),
    "fullOOF_all": dict(gas=(yg, fg, M, np.ones(len(yg))), eui=(ye, fe, M, A)),
}
out = {"test_band_counts": {m: np.histogram(test_pg[m], EDGES)[0].tolist() for m in ("gas", "eui")},
       "edges": EDGES[:-1].tolist()}
rng = np.random.default_rng(5)
for name, d in src.items():
    r = {}
    tot_direct, tot_rw = 0.0, 0.0
    boots = np.zeros(300)
    for m in ("gas", "eui"):
        y, p, mm, sc = d[m]
        pg = p * sc
        pn = (pg + CL[f"c_{m}"] * adj(pg, mm)) / sc
        de = (y - pn) ** 2 - (y - p) ** 2
        rmse = np.sqrt(np.mean((y - p) ** 2))
        b = np.clip(np.searchsorted(EDGES, pg, side="right") - 1, 0, len(EDGES) - 2)
        share_t = np.histogram(test_pg[m], EDGES)[0] / len(sub)
        cnt = np.bincount(b, minlength=len(EDGES) - 1)
        mean_b = np.bincount(b, weights=de, minlength=len(EDGES) - 1) / np.maximum(cnt, 1)
        direct = (np.sqrt(np.mean((y - pn) ** 2)) - rmse) / SIG[m] / 5
        rw = float(np.sum(share_t * mean_b) / (2 * rmse * SIG[m]) / 5)
        for i in range(300):
            ib = rng.integers(0, len(y), len(y))
            cb = np.bincount(b[ib], minlength=len(EDGES) - 1)
            mb = np.bincount(b[ib], weights=de[ib], minlength=len(EDGES) - 1) / np.maximum(cb, 1)
            boots[i] += np.sum(share_t * mb) / (2 * rmse * SIG[m]) / 5
        r[m] = dict(direct=float(direct), reweighted_to_test=rw, src_band_counts=cnt.tolist(),
                    src_band_mean_de=mean_b.round(0).tolist())
        tot_direct += direct
        tot_rw += rw
    r["total_direct"] = float(tot_direct)
    r["total_reweighted_to_test"] = float(tot_rw)
    r["reweighted_boot_sd"] = float(boots.std())
    r["reweighted_boot_frac_worse"] = float((boots > 0).mean())
    out[name] = r
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
