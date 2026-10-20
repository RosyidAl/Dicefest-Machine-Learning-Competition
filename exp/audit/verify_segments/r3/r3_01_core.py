"""verify_segments r3 (third, independent re-derivation): truncated-normal (inverse Mills ratio) cap calibration of
gas / EUI near the gas < 1e6 truncation.  Everything is recomputed from exp/audit/chain_H.npz + cache/train.parquet
(label columns only) with my own code (no import from the auditor's scripts).

cal(g; k) = g - s * phi(a) / Phi(a),  s = k g,  a = (1e6 - g) / s   (applied where g > 6e5, as in the claim)
EUI: g = eui * A,  eui' = eui * cal(g) / g.

Honesty: k is fitted on rows whose labels are never used for scoring:
  (a) P-side teacher (P-OOF VC blend, models trained on P only) -> scored on H corrected (the chain analog of the
      submission; lambda of the residual correction 5-fold cross-fitted on H);
  (b) cross-fits inside H (5-fold x 20, 2-fold x 100, 10 contiguous sub_n blocks left out one at a time);
  (c) reverse direction: k fitted on all of H corrected -> scored on the P teacher (rows independent of H).
A double bootstrap (resample P rows -> refit k, resample H rows -> score) gives the uncertainty incl. the k-fit.
All effects are TOTAL-score changes = per-target NRMSE change / 5 (negative = better).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/r3/r3_01_core.py \
         > exp/audit/verify_segments/r3/r3_01_core.log 2>&1
"""
import json
import resource
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

T0 = time.time()
SIG = {"gas": 151750.04, "eui": 40.65}
CAP, THR = 1e6, 6e5
L2PI = 0.5 * np.log(2 * np.pi)
OUT = "exp/audit/verify_segments/r3"

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
tr = pd.read_parquet("cache/train.parquet",
                     columns=["target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm", "bld_dim_06"])
yg_all, ye_all = tr["target_macro_gas_kwh"].values.astype(float), tr["target_eui_gas_kwh_per_sqm"].values.astype(float)
assert np.array_equal(A, tr["bld_dim_06"].values.astype(float))
del tr
AP, AH = A[P], A[H]

# ---------------------------------------------------------------- label / split sanity
chk = {
    "P_H_disjoint": bool(np.intersect1d(P, H).size == 0), "P_plus_H": int(P.size + H.size),
    "gas_y_H_vs_parquet": float(np.max(np.abs(z["gas_y"] - yg_all[H]))),
    "gas_pP_plus_rP_vs_parquet": float(np.max(np.abs(z["gas_pP"] + z["gas_rP"] - yg_all[P]))),
    "eui_y_H_vs_parquet": float(np.max(np.abs(z["eui_y"] - ye_all[H]))),
    "eui_pP_plus_rP_vs_parquet": float(np.max(np.abs(z["eui_pP"] + z["eui_rP"] - ye_all[P]))),
    "train_gas_max": float(yg_all.max()), "train_gas_n_ge_1e6": int((yg_all >= CAP).sum()),
    "eui_times_A_vs_gas_max_rel": float(np.max(np.abs(ye_all * A - yg_all) / yg_all)),
}
print("sanity", json.dumps(chk))


def cal(g, k):
    """truncated-normal mean below CAP, applied where g > THR (exactly the claimed mask)."""
    g = np.asarray(g, dtype=float)
    out = g.copy()
    m = g > THR
    s = k * g[m]
    a = (CAP - g[m]) / s
    out[m] = g[m] - s * np.exp(-0.5 * a * a - L2PI - log_ndtr(a))
    return out


def cal_eui(e, Ae, k):
    g = e * Ae
    gn = cal(g, k)
    return np.where(g > THR, e * gn / np.where(g > THR, g, 1.0), e)


def nr(err, m):
    return float(np.sqrt(np.mean(err * err)) / SIG[m])


def fit_k(y, p, kind="gas", Ae=None, w=None):
    """least-squares k on the given rows (only rows above THR matter); grid 0.0025 then bounded refinement"""
    if kind == "gas":
        m = p > THR
        yy, pp = y[m], p[m]
        f = lambda k: float(np.sum((yy - cal(pp, k)) ** 2 * (1 if w is None else w[m])))  # noqa: E731
    else:
        m = p * Ae > THR
        yy, pp, aa = y[m], p[m], Ae[m]
        f = lambda k: float(np.sum((yy - cal_eui(pp, aa, k)) ** 2 * (1 if w is None else w[m])))  # noqa: E731
    grid = np.arange(0.005, 0.3001, 0.0025)
    v = [f(k) for k in grid]
    k0 = grid[int(np.argmin(v))]
    r = minimize_scalar(f, bounds=(max(0.001, k0 - 0.003), k0 + 0.003), method="bounded",
                        options=dict(xatol=1e-6))
    return float(r.x)


def dtot(y, p0, p1, m):
    return (nr(y - p1, m) - nr(y - p0, m)) / 5


# ---------------------------------------------------------------- views
yP_g, pP_g = z["gas_pP"] + z["gas_rP"], z["gas_pP"]
yP_e, pP_e = z["eui_pP"] + z["eui_rP"], z["eui_pP"]
yH_g, tH_g, cH_g = z["gas_y"], z["gas_teacher"], z["gas_corrected"]
yH_e, tH_e, cH_e = z["eui_y"], z["eui_teacher"], z["eui_corrected"]
base = dict(gas=nr(yH_g - cH_g, "gas"), eui=nr(yH_e - cH_e, "eui"),
            gas_teacher=nr(yH_g - tH_g, "gas"), eui_teacher=nr(yH_e - tH_e, "eui"))
print("base NRMSE on H", {k: round(v, 6) for k, v in base.items()})

res = {}
# (a) P teacher -> H
kP_g = fit_k(yP_g, pP_g)
kP_e = fit_k(yP_e, pP_e, "eui", AP)
res["a_kP_gas"], res["a_kP_eui_own"] = kP_g, kP_e
res["a_H_gas"] = dtot(yH_g, cH_g, cal(cH_g, kP_g), "gas")
res["a_H_eui_gask"] = dtot(yH_e, cH_e, cal_eui(cH_e, AH, kP_g), "eui")
res["a_H_eui_ownk"] = dtot(yH_e, cH_e, cal_eui(cH_e, AH, kP_e), "eui")
res["a_H_total_gask"] = res["a_H_gas"] + res["a_H_eui_gask"]
res["a_Hteacher_gas"] = dtot(yH_g, tH_g, cal(tH_g, kP_g), "gas")
res["a_Hteacher_eui"] = dtot(yH_e, tH_e, cal_eui(tH_e, AH, kP_g), "eui")
# the exact claimed constant k = 0.08
res["k008_H_gas"] = dtot(yH_g, cH_g, cal(cH_g, 0.08), "gas")
res["k008_H_eui"] = dtot(yH_e, cH_e, cal_eui(cH_e, AH, 0.08), "eui")
res["k008_H_total"] = res["k008_H_gas"] + res["k008_H_eui"]
# in-sample (diagnostic only, not a claim): k fitted on H corrected and on H teacher
res["diag_kH_corrected_gas"] = fit_k(yH_g, cH_g)
res["diag_kH_teacher_gas"] = fit_k(yH_g, tH_g)
res["diag_kH_corrected_eui"] = fit_k(yH_e, cH_e, "eui", AH)
# (c) reverse: all-H fit -> P teacher
kH_g = res["diag_kH_corrected_gas"]
res["c_P_gas"] = dtot(yP_g, pP_g, cal(pP_g, kH_g), "gas")
res["c_P_eui_gask"] = dtot(yP_e, pP_e, cal_eui(pP_e, AP, kH_g), "eui")
for k, v in res.items():
    print(f"  {k:24s} {v:+.8f}" if abs(v) < 0.01 else f"  {k:24s} {v:.5f}")


# ---------------------------------------------------------------- (b) cross-fits inside H
def crossfit(splits):
    pg, pe = cH_g.copy(), cH_e.copy()
    ks = []
    for a, b in splits:
        kg = fit_k(yH_g[a], cH_g[a])
        ke = fit_k(yH_e[a], cH_e[a], "eui", AH[a])
        pg[b] = cal(cH_g[b], kg)
        pe[b] = cal_eui(cH_e[b], AH[b], ke)
        ks.append((kg, ke))
    return dtot(yH_g, cH_g, pg, "gas"), dtot(yH_e, cH_e, pe, "eui"), ks


def kfold(n, K, seed):
    perm = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(perm, K)
    return [(np.setdiff1d(perm, parts[i]), parts[i]) for i in range(K)]


cf = {}
for name, K, reps in (("5fold", 5, 20), ("2fold", 2, 60)):
    g_, e_ = [], []
    for r in range(reps):
        dg, de, _ = crossfit(kfold(len(H), K, 9000 + r))
        g_.append(dg)
        e_.append(de)
    g_, e_ = np.array(g_), np.array(e_)
    cf[name] = dict(gas_mean=g_.mean(), gas_sd=g_.std(), gas_worst=g_.max(), eui_mean=e_.mean(), eui_sd=e_.std(),
                    tot_mean=(g_ + e_).mean(), tot_worst=(g_ + e_).max(), share_tot_neg=float(np.mean(g_ + e_ < 0)))
# contiguous sub_n blocks (spatially ordered ids) left out one at a time
sH = subn[H]
edges = np.quantile(sH, np.linspace(0, 1, 11))
blk = np.clip(np.searchsorted(edges, sH, side="right") - 1, 0, 9)
dg, de, ks = crossfit([(np.where(blk != b)[0], np.where(blk == b)[0]) for b in range(10)])
cf["subn_blocks"] = dict(gas=dg, eui=de, tot=dg + de, k_gas=[round(k[0], 4) for k in ks],
                         k_eui=[round(k[1], 4) for k in ks])
print("cross-fits inside H:")
for k, v in cf.items():
    print(f"  {k:12s}", {kk: (round(vv, 8) if isinstance(vv, float) else vv) for kk, vv in v.items()})

# ---------------------------------------------------------------- double bootstrap (k refit on resampled P rows)
rng = np.random.default_rng(123)
mP = pP_g > THR * 0.999
yPm, pPm = yP_g[mP], pP_g[mP]
e0g, e0e = (yH_g - cH_g) ** 2, (yH_e - cH_e) ** 2
bs = {"H": [], "public": [], "private": []}
sizes = {"H": len(H), "public": 55196, "private": 128791}
kb = []
for it in range(400):
    w = rng.poisson(1.0, size=mP.sum()).astype(float)
    kg = fit_k(yPm, pPm, w=w)
    kb.append(kg)
    e1g = (yH_g - cal(cH_g, kg)) ** 2
    e1e = (yH_e - cal_eui(cH_e, AH, kg)) ** 2
    for nm, sz in sizes.items():
        for _ in range(3):
            i = rng.integers(0, len(H), sz)
            d = ((np.sqrt(e1g[i].mean()) - np.sqrt(e0g[i].mean())) / SIG["gas"]
                 + (np.sqrt(e1e[i].mean()) - np.sqrt(e0e[i].mean())) / SIG["eui"]) / 5
            bs[nm].append(d)
print(f"double bootstrap: k (P resampled) mean {np.mean(kb):.4f} sd {np.std(kb):.4f} "
      f"5..95% {np.quantile(kb, 0.05):.4f}..{np.quantile(kb, 0.95):.4f}")
for nm, v in bs.items():
    v = np.array(v)
    print(f"  {nm:8s} n={sizes[nm]:6d}: total change mean {v.mean():+.8f} sd {v.std():.8f}  P(<0) {np.mean(v < 0):.3f}"
          f"  5% {np.quantile(v, 0.05):+.8f}  95% {np.quantile(v, 0.95):+.8f}")

# ---------------------------------------------------------------- concentration of the H gain (P-fit k)
p1 = cal(cH_g, kP_g)
dse = (yH_g - p1) ** 2 - (yH_g - cH_g) ** 2
o = np.argsort(dse)
tot = dse.sum()
print(f"gas SSE change on H (sigma^2 units) {tot / SIG['gas'] ** 2:+.3f}; rows changed > 1 kWh "
      f"{int((np.abs(p1 - cH_g) > 1).sum())}; helped {int((dse < -1e6).sum())} hurt {int((dse > 1e6).sum())}")
for kdrop in (0, 1, 3, 5, 10, 20):
    keep = np.ones(len(H), bool)
    keep[o[:kdrop]] = False
    sse0, sse1 = np.sum(((yH_g - cH_g) ** 2)), np.sum(((yH_g - cH_g) ** 2)) + dse[keep].sum()
    print(f"   remove gain of the {kdrop:2d} most-helped rows: gas dTotal "
          f"{(np.sqrt(sse1 / len(H)) - np.sqrt(sse0 / len(H))) / SIG['gas'] / 5:+.8f}")
# same for the hurt side
for kdrop in (5, 10):
    keep = np.ones(len(H), bool)
    keep[o[::-1][:kdrop]] = False
    sse0 = np.sum((yH_g - cH_g) ** 2)
    sse1 = sse0 + dse[keep].sum()
    print(f"   remove loss of the {kdrop:2d} most-hurt rows: gas dTotal "
          f"{(np.sqrt(sse1 / len(H)) - np.sqrt(sse0 / len(H))) / SIG['gas'] / 5:+.8f}")

json.dump(dict(sanity=chk, base=base, res=res, crossfit=cf, kP_gas=kP_g,
               boot={k: [float(np.mean(v)), float(np.std(v)), float(np.mean(np.array(v) < 0))] for k, v in bs.items()}),
          open(f"{OUT}/r3_01_core.json", "w"), indent=1, default=float)
print(f"elapsed {time.time() - T0:.1f} s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
