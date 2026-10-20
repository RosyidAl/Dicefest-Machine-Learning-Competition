"""verify2_segments: transfer of the gas-cap gain to the TEST level distribution.

The gain is concentrated in the few rows with the highest predictions, and the test (full-data, fold-averaged models)
has fewer extreme predictions than the H chain (P-only models). Here the per-row squared-error change at k = 0.08 is
averaged within prediction-level bins on two honest sources, then re-weighted with the test rows' level counts:
  source 1: H chain corrected predictions (models trained on P only, correction cross-fitted on H)
  source 2: full-train OOF varying-coefficient blend A2xM2xp2 (models trained on 90% of train; my Gram NNLS, as v3)
k is fixed at 0.08 here (cross-fitted k gave the same, see v2 / v3). Bootstrap over source rows within bins gives sd.
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.special import log_ndtr

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402

GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
CAP, K = 1e6, 0.08
OUT = "exp/audit/verify2_segments"
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]


def mills(a):
    return np.exp(-0.5 * a * a - 0.5 * np.log(2.0 * np.pi) - log_ndtr(a))


def calib(g, k=K):
    out = np.array(g, dtype=float, copy=True)
    m = out > 5e5
    s = k * out[m]
    out[m] = out[m] - s * mills((CAP - out[m]) / s)
    return out


# full-OOF VC blend (cached)
cache = f"{OUT}/fulloof_vc.npz"
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12"])
yg, ye = tr[GAS].values.astype(float), tr[EUI].values.astype(float)
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
n = len(yg)
if not os.path.exists(cache):
    names = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
    Pf = np.column_stack([np.load(f"exp/oof/{nm}.npy") for nm in names])

    def gram_nnls(cols, y, s, chunk=40000):
        G, c = None, None
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(n, i0 + chunk))
            X = cols(sl) * s[sl, None]
            G = X.T @ X if G is None else G + X.T @ X
            c = X.T @ (y[sl] * s[sl]) if c is None else c + X.T @ (y[sl] * s[sl])
        d = np.sqrt(np.diag(G))
        L = np.linalg.cholesky(G / np.outer(d, d) + 1e-12 * np.eye(len(d)))
        w, _ = nnls(L.T, solve_triangular(L, c / d, lower=True), maxiter=50 * len(d))
        return w / d

    res = {}
    for metric in ("gas", "eui"):
        s = A if metric == "gas" else np.ones(n)
        w = nnls(Pf * s[:, None], ye * s)[0]
        p = Pf @ w
        B = vc_basis("A2xM2xp2", dict(A=A, M=M, p=p), dict(A=A, M=M, p=p))
        nb, nm = B.shape[1], Pf.shape[1]
        W = gram_nnls(lambda sl: np.concatenate([Pf[sl] * B[sl, [j]] for j in range(nb)], axis=1), ye, s)
        res[metric] = ((Pf @ W.reshape(nb, nm).T) * B).sum(axis=1)
    np.savez(cache, gas=res["gas"] * A, eui=res["eui"])
    del Pf
fo = np.load(cache)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
AH = z["A"][z["H"]]
sources = {
    "H chain corrected": dict(gas=(z["gas_y"], z["gas_corrected"]), eui=(z["eui_y"], z["eui_corrected"], AH)),
    "full-OOF VC": dict(gas=(yg, fo["gas"]), eui=(ye, fo["eui"], A)),
}
# test levels (non-duplicate rows; duplicates keep train values)
base = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
te = pd.read_parquet("cache/test.parquet", columns=KEY)
trk = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS]).groupby(KEY, sort=False)[GAS].first().reset_index()
dup = ~te.merge(trk, on=KEY, how="left")[GAS].isna().values
At = te["bld_dim_06"].values.astype(float)
lev_t = {"gas": base[GAS].values[~dup], "eui": (base[EUI].values * At)[~dup]}
NT = len(base)
edges = np.array([5e5, 8e5, 8.5e5, 9e5, 9.25e5, 9.5e5, 9.75e5, 1e6, 1.05e6, np.inf])
nrm0 = {"gas": 0.2451, "eui": 0.6486}  # chain-level NRMSE; only sets the sqrt linearisation
rng = np.random.default_rng(11)
for sname, src in sources.items():
    tot_mean, tot_boot = 0.0, np.zeros(2000)
    for metric in ("gas", "eui"):
        if metric == "gas":
            y, p = src["gas"]
            q, lev = calib(p), p
        else:
            y, p, area = src["eui"]
            g = p * area
            q, lev = p * calib(g) / g, g
        dse = (y - q) ** 2 - (y - p) ** 2
        tcount = np.histogram(lev_t[metric], edges)[0]
        dsum, boots = 0.0, np.zeros(2000)
        rows = []
        for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mm = np.flatnonzero((lev >= lo) & (lev < hi))
            mu = dse[mm].mean() if len(mm) else 0.0
            dsum += tcount[b] * mu
            if len(mm):
                bm = dse[mm][rng.integers(0, len(mm), (2000, len(mm)))].mean(axis=1)
                boots += tcount[b] * bm
            rows.append(f"[{lo / 1e6:.3f},{hi / 1e6:.3f}) src n {len(mm):5d} test n {tcount[b]:4d} mean dSE/sig2 "
                        f"{mu / SIG[metric] ** 2:+.4f}")
        mse0 = (nrm0[metric] * SIG[metric]) ** 2

        def to_tot(x):
            return (np.sqrt(mse0 + x / NT) - np.sqrt(mse0)) / SIG[metric] / 5

        own = (np.sqrt(np.mean((y - q) ** 2)) - np.sqrt(np.mean((y - p) ** 2))) / SIG[metric] / 5
        print(f"\n{sname} / {metric}: own-rows dTotal {own:+.8f}; re-weighted to test levels {to_tot(dsum):+.8f}"
              f" (bootstrap sd {np.std(to_tot(boots)):.8f})")
        for r in rows:
            print("   " + r)
        tot_mean += to_tot(dsum)
        tot_boot += to_tot(boots)
    print(f"{sname}: gas+eui expected test dTotal {tot_mean:+.8f}  sd {tot_boot.std():.8f}  P(<0) {np.mean(tot_boot < 0):.3f}")
