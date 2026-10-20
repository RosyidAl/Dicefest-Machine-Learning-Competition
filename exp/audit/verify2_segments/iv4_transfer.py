"""Independent re-verification (round 'iv'): expected change of the TOTAL score on the actual test rows of the candidate
(k = 0.08 calibration of sub_v19vce gas / eui), using a nearest-level conditional distribution of the truth.

For every non-duplicate test row with level g = pred (gas) or pred * A (eui) above 6e5, the truth is drawn from the K source
rows whose prediction level is nearest in log scale. Sources: (a) holdout chain H corrected (models trained on P only),
(b) full-train OOF of my global NNLS blend of the 27 gas members (models trained on 90% folds). The same draw Y (gas units)
is used for gas and for eui (eui truth = Y / A_test), since eui = gas / A exactly.
Public / private: random 30% / 70% row subsets of test, 4000 simulations each.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_segments/iv4_transfer.py
"""
import resource
import time

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.special import erfcx

t0 = time.time()
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
NR = {"gas": 0.2451, "eui": 0.6486}  # NRMSE level of the base on H (only scales the conversion SSE -> NRMSE)
CAP, K0 = 1e6, 0.08


def fnew(g, k=K0):
    s = k * g
    a = (CAP - g) / s
    return g - s * np.sqrt(2.0 / np.pi) / erfcx(-a / np.sqrt(2.0))


bld = [f"bld_dim_{i:02d}" for i in range(1, 19)]
tr = pd.read_parquet("cache/train.parquet", columns=["subzone_id", "socio_demog_005"] + bld + [GAS, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id", "subzone_id", "socio_demog_005"] + bld)
key = ["subzone_id", "socio_demog_005"] + bld
m_ = te[key].merge(tr.groupby(key, sort=False)[GAS].first().reset_index(), on=key, how="left")
isdup = ~np.isnan(m_[GAS].values)
base = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (base["id"].values == te["id"].values).all()
At = te["bld_dim_06"].values
gt = base[GAS].values.astype(float)
gEt = base[EUI].values.astype(float) * At
live = ~isdup
selg = live & (gt > 6e5)
selE = live & (gEt > 6e5)
n_te = len(te)
print(f"test rows {n_te}; dup {isdup.sum()}; live rows with gas level > 6e5: {selg.sum()}, eui level > 6e5: {selE.sum()}")

# sources
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H = z["H"]
srcH = (z["gas_corrected"], z["gas_y"])
names = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
A = tr["bld_dim_06"].values
X = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
wg, _ = nnls(X * A[:, None], tr[GAS].values)
srcF = (X @ wg * A, tr[GAS].values)
del X
edges = [0.6, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, np.inf]
print("level distribution per 100k rows (gas):")
for nm, lv, n in (("test (live)", gt[live], live.sum()), ("H corrected", srcH[0], len(H)), ("full OOF", srcF[0], len(srcF[0]))):
    c = np.histogram(lv / 1e6, edges)[0]
    print(f"   {nm:12s}: " + "  ".join(f"[{a:.2f},{b:.2f}) {1e5 * x / n:6.1f}" for a, b, x in zip(edges[:-1], edges[1:], c)))


def neighbours(src_level, q_level, K):
    """indices of the K source rows nearest in log level for each query level"""
    o = np.argsort(src_level)
    sl = np.log(src_level[o])
    ql = np.log(q_level)
    pos = np.searchsorted(sl, ql)
    out = np.empty((len(q_level), K), dtype=np.int64)
    for i, (p, v) in enumerate(zip(pos, ql)):
        lo, hi = max(p - K, 0), min(p + K, len(sl))
        cand = np.arange(lo, hi)
        out[i] = o[cand[np.argsort(np.abs(sl[cand] - v), kind="stable")[:K]]]
    return out


def dtot_from_D(D, met):
    return (np.sqrt(NR[met] ** 2 + D) - NR[met]) / 5


ig = np.where(selg | selE)[0]  # rows that can change in either column
g_q = gt[ig]
gE_q = gEt[ig]
dg_mask = selg[ig]
dE_mask = selE[ig]
gnew = np.where(dg_mask, fnew(np.where(dg_mask, g_q, 1.0)), g_q)
gEnew = np.where(dE_mask, fnew(np.where(dE_mask, gE_q, 1.0)), gE_q)
Aq = At[ig]
for sname, (sl, sy) in (("H corrected", srcH), ("full OOF NNLS", srcF)):
    for K in (10, 20, 40):
        nb = neighbours(sl, g_q, K)  # neighbours by the gas level (eui level is within a few % of it)
        Y = sy[nb]  # (rows, K) truth draws, gas units
        dse_g = ((Y - gnew[:, None]) ** 2 - (Y - g_q[:, None]) ** 2).mean(1) / SIG["gas"] ** 2
        dse_e = (((Y - gEnew[:, None]) ** 2 - (Y - gE_q[:, None]) ** 2) / Aq[:, None] ** 2).mean(1) / SIG["eui"] ** 2
        Dg, De = dse_g.sum() / n_te, dse_e.sum() / n_te
        eg, ee = dtot_from_D(Dg, "gas"), dtot_from_D(De, "eui")
        line = f"{sname:14s} K {K:2d}: expected full-test dTotal gas {eg:+.8f} eui {ee:+.8f} sum {eg + ee:+.8f}"
        if K == 20:
            rng = np.random.default_rng(2468 if sname.startswith("H") else 1357)
            sims = {"public": [], "private": []}
            for _ in range(4000):
                pub = rng.random(len(ig)) < 0.3  # each test row is public with prob 0.3
                pick = Y[np.arange(len(ig)), rng.integers(0, K, len(ig))]
                sg = ((pick - gnew) ** 2 - (pick - g_q) ** 2) / SIG["gas"] ** 2
                se = (((pick - gEnew) ** 2 - (pick - gE_q) ** 2) / Aq ** 2) / SIG["eui"] ** 2
                for nm, msk, n in (("public", pub, 55196), ("private", ~pub, 128791)):
                    sims[nm].append(dtot_from_D(sg[msk].sum() / n, "gas") + dtot_from_D(se[msk].sum() / n, "eui"))
            for nm, v in sims.items():
                v = np.array(v)
                line += f"\n      {nm:7s} sim: mean {v.mean():+.8f} sd {v.std():.8f} P(improve) {(v < 0).mean():.4f} " \
                        f"P(<= -1e-5) {(v <= -1e-5).mean():.4f} 5%..95% {np.quantile(v, 0.05):+.8f}..{np.quantile(v, 0.95):+.8f}"
        print(line)
# per-row view of the biggest test changes (H corrected source, K 20)
nb = neighbours(srcH[0], g_q, 20)
Y = srcH[1][nb]
dse_g = ((Y - gnew[:, None]) ** 2 - (Y - g_q[:, None]) ** 2).mean(1) / SIG["gas"] ** 2
o = np.argsort(dse_g)[:12]
print("12 test rows with the largest expected gas gain (H source, K 20): pred -> new, E[truth], expected dSE/sigma^2")
for i in o:
    print(f"   {g_q[i]:10.0f} -> {gnew[i]:10.0f}  E[y] {Y[i].mean():10.0f}  dSE {dse_g[i]:+.4f}")
print(f"share of the expected gas gain from these 12 rows: {dse_g[o].sum() / dse_g.sum():.3f}")
print(f"\nelapsed {time.time() - t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
