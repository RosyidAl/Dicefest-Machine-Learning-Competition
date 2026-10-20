"""verify2_integrity t2r_01: independent reproduction of the integrity-lens claim
"gas is truncated at 1e6 kWh; a truncated-normal shrink of gas and EUI near the bound gains ~-2.56e-5 total".

Own code (does not import or reuse exp/audit/integrity/*). Input: exp/audit/chain_H.npz only (no raw data).
Steps
  1. facts: train gas max / histogram near 1e6 (P labels = pP + rP, H labels = gas_y); EUI*A vs gas identity on H.
  2. bias near the bound by prediction bin: H corrected, H teacher, P-side fitted blend.
  3. own truncated-normal shrink  p' = p + c * shift(p, k*s),  shift = E[y | y < U] - p for y ~ N(p, s^2),
     s = |p| sqrt(a + b/M); EUI is shrunk on the gas scale (p_eui * A) and divided back.
     Honest schemes (never in-sample):
       A  claim replication: a, b, k, c fitted on P (P-side blend), scored on H corrected.
       B  cross-fit inside H: repeated KFold (5-fold x 10 seeds and 2-fold x 10 seeds); a, b fitted on P;
          k (grid) and c chosen on the fit folds of H, scored on the held-out fold.
       C  spatial blocks of H: 4 contiguous sub_n blocks, leave-one-block-out (fit k, c on 3 blocks).
       D  reverse direction: k, c fitted on H corrected, scored on the P-side blend (pP).
  4. paired bootstrap of the H gain for scheme A; gain decomposition by prediction band; hard-clip comparison.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/t2r_01_repro.py
"""
import json

import numpy as np
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

OUT = "exp/audit/verify2_integrity/t2r_01_repro.json"
SIG = {"gas": 151750.04, "eui": 40.65}
U = 1e6
KGRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
subnH = z["subn"][H]
res = {}


def logphi(x):
    return -0.5 * x * x - 0.5 * np.log(2 * np.pi)


def shift(pg, s):
    """E[y | y < U] - mu for y ~ N(mu = pg, s^2), computed in logs (stable when pg >> U)"""
    a = (U - pg) / s
    return -s * np.exp(logphi(a) - log_ndtr(a))


def nrmse(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


# ---------------------------------------------------------------- 1. facts
gas_all = np.full(len(A), np.nan)
gas_all[P] = z["gas_pP"] + z["gas_rP"]
gas_all[H] = z["gas_y"]
edges = [8e5, 8.5e5, 9e5, 9.25e5, 9.5e5, 9.75e5, 1e6, 1.025e6, 1.05e6, 1.1e6, np.inf]
hist = {f"{lo:.0f}-{hi:.0f}": int(((gas_all >= lo) & (gas_all < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])}
res["facts"] = dict(n=int(np.isfinite(gas_all).sum()), gas_max=float(np.nanmax(gas_all)),
                    n_ge_1e6=int((gas_all >= 1e6).sum()), n_ge_999k=int((gas_all >= 9.99e5).sum()), hist=hist,
                    eui_times_A_vs_gas_H_max_rel=float(np.max(np.abs(z["eui_y"] * AH - z["gas_y"]) /
                                                              np.maximum(z["gas_y"], 1))))
print("facts", json.dumps(res["facts"]))

# ---------------------------------------------------------------- 2. bias by prediction bin
def bias_table(y, p):
    out = {}
    for lo, hi in ((7e5, 8e5), (8e5, 8.5e5), (8.5e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, np.inf), (9e5, np.inf),
                   (9.5e5, np.inf)):
        g = (p >= lo) & (p < hi)
        r = y[g] - p[g]
        out[f"{lo:.0f}-{hi:.0f}"] = dict(n=int(g.sum()), mean_resid=float(r.mean()) if g.any() else None,
                                         se=float(r.std(ddof=1) / np.sqrt(g.sum())) if g.sum() > 1 else None)
    return out


pc = {"gas": np.clip(z["gas_corrected"], 0, None), "eui": np.clip(z["eui_corrected"], 0, None)}
res["bias"] = dict(
    gas_H_corrected=bias_table(z["gas_y"], pc["gas"]),
    gas_H_teacher=bias_table(z["gas_y"], z["gas_teacher"]),
    gas_P_fitted=bias_table(z["gas_pP"] + z["gas_rP"], z["gas_pP"]),
    euiA_H_corrected=bias_table(z["eui_y"] * AH, pc["eui"] * AH),
    euiA_P_fitted=bias_table((z["eui_pP"] + z["eui_rP"]) * AP, z["eui_pP"] * AP))
for k in ("gas_H_corrected", "gas_P_fitted", "euiA_H_corrected"):
    t = res["bias"][k]
    print("bias", k, {b: (v["n"], None if v["mean_resid"] is None else round(v["mean_resid"])) for b, v in t.items()})

# ---------------------------------------------------------------- 3. own shrink
# variance model on P: (r/p)^2 = a + b/M (OLS), separately from gas and from EUI*A residuals (same quantity)
def var_model(p, r, Mv):
    g = p > 1000
    X = np.column_stack([np.ones(g.sum()), 1 / Mv[g]])
    return np.linalg.lstsq(X, (r[g] / p[g]) ** 2, rcond=None)[0]


ab_gas = var_model(z["gas_pP"], z["gas_rP"], MP)
ab_euiA = var_model(z["eui_pP"] * AP, z["eui_rP"] * AP, MP)
res["var_model"] = dict(gas=ab_gas.tolist(), euiA=ab_euiA.tolist())
print("var model (r/p)^2 = a + b/M:", res["var_model"])


def xcol(m, p, Ar, Mv, k, ab):
    """shift column on the metric's own scale; p is the metric prediction (gas, or EUI)"""
    pg = p * Ar if m == "eui" else p
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(ab[0] + ab[1] / Mv, 1e-6)), 1.0)
    x = shift(pg, s)
    return x / Ar if m == "eui" else x


def fit_c(x, r):
    den = float(x @ x)
    return float(x @ r / den) if den > 0 else 0.0


def fit_kc(m, p, y, Ar, Mv, ab):
    best = None
    for k in KGRID:
        x = xcol(m, p, Ar, Mv, k, ab)
        c = fit_c(x, y - p)
        sse = float(np.sum((y - p - c * x) ** 2))
        if best is None or sse < best[2]:
            best = (k, c, sse)
    return best[0], best[1]


data = {}
for m in ("gas", "eui"):
    data[m] = dict(pP=z[f"{m}_pP"], yP=z[f"{m}_pP"] + z[f"{m}_rP"], pH=pc[m], yH=z[f"{m}_y"])
AB = {"gas": ab_gas, "eui": ab_gas}  # claim uses one (a, b) for both (gas-scale relative variance)

# ---- scheme A: fit on P, score on H (k grid on P)
schemeA = {}
for m in ("gas", "eui"):
    d = data[m]
    ArP, ArH = (AP, AH) if m == "eui" else (1.0, 1.0)
    grid = {}
    for k in KGRID:
        xP = xcol(m, d["pP"], ArP, MP, k, AB[m])
        c = fit_c(xP, d["yP"] - d["pP"])
        sseP = float(np.sum((d["yP"] - d["pP"] - c * xP) ** 2))
        xH = xcol(m, d["pH"], ArH, MH, k, AB[m])
        dt = (nrmse(d["yH"], d["pH"] + c * xH, m) - nrmse(d["yH"], d["pH"], m)) / 5
        grid[k] = dict(c=c, sseP=sseP, H_dtotal=dt)
    kb = min(grid, key=lambda k: grid[k]["sseP"])
    schemeA[m] = dict(grid=grid, k_P=kb, c_P=grid[kb]["c"], H_dtotal=grid[kb]["H_dtotal"],
                      base_H=nrmse(d["yH"], d["pH"], m))
schemeA["total"] = schemeA["gas"]["H_dtotal"] + schemeA["eui"]["H_dtotal"]
res["A_fitP_scoreH"] = schemeA
print("A fit P -> H:", {m: (schemeA[m]["k_P"], round(schemeA[m]["c_P"], 4), round(schemeA[m]["H_dtotal"] * 1e5, 3))
                        for m in ("gas", "eui")}, "total e-5", round(schemeA["total"] * 1e5, 3))
print("  grid gas (k: c, dtotal e-5)", {k: (round(v["c"], 3), round(v["H_dtotal"] * 1e5, 3))
                                         for k, v in schemeA["gas"]["grid"].items()})
print("  grid eui (k: c, dtotal e-5)", {k: (round(v["c"], 3), round(v["H_dtotal"] * 1e5, 3))
                                         for k, v in schemeA["eui"]["grid"].items()})

# ---- scheme B: cross-fit within H (k and c chosen on the fit folds)
def crossfit(m, splits, fixed_k=None):
    d = data[m]
    y, p = d["yH"], d["pH"]
    Ar = AH if m == "eui" else np.ones(len(H))
    pn = p.copy()
    cs = []
    for a, b in splits:
        if fixed_k is None:
            k, c = fit_kc(m, p[a], y[a], Ar[a], MH[a], AB[m])
        else:
            k = fixed_k
            c = fit_c(xcol(m, p[a], Ar[a], MH[a], k, AB[m]), y[a] - p[a])
        pn[b] = p[b] + c * xcol(m, p[b], Ar[b], MH[b], k, AB[m])
        cs.append((k, c))
    return (nrmse(y, pn, m) - nrmse(y, p, m)) / 5, cs


schemeB = {}
for nf in (5, 2):
    for fk in (None, 1.0, 0.7):
        tot = []
        per = {"gas": [], "eui": []}
        cs_all = {"gas": [], "eui": []}
        for seed in range(10):
            t = 0.0
            for m in ("gas", "eui"):
                dt, cs = crossfit(m, list(KFold(nf, shuffle=True, random_state=100 + seed).split(H)), fk)
                per[m].append(dt)
                cs_all[m] += cs
                t += dt
            tot.append(t)
        key = f"{nf}fold_k{'grid' if fk is None else fk}"
        schemeB[key] = dict(total_mean=float(np.mean(tot)), total_sd_over_seeds=float(np.std(tot)),
                            total_min=float(np.min(tot)), total_max=float(np.max(tot)),
                            gas_mean=float(np.mean(per["gas"])), eui_mean=float(np.mean(per["eui"])),
                            frac_negative=float(np.mean(np.array(tot) < 0)),
                            k_c_range={m: [min(c for _, c in cs_all[m]), max(c for _, c in cs_all[m])]
                                       for m in ("gas", "eui")},
                            k_chosen={m: sorted(set(k for k, _ in cs_all[m])) for m in ("gas", "eui")})
        print("B", key, {k: (round(v * 1e5, 3) if isinstance(v, float) else v) for k, v in schemeB[key].items()})
res["B_crossfit_H"] = schemeB

# ---- scheme C: 4 contiguous sub_n blocks of H, leave-one-block-out
q = np.quantile(subnH, [0.25, 0.5, 0.75])
blk = np.searchsorted(q, subnH, side="right")
splits = [(np.where(blk != g)[0], np.where(blk == g)[0]) for g in range(4)]
schemeC = {}
for fk in (None, 0.7, 1.0):
    t = 0.0
    det = {}
    for m in ("gas", "eui"):
        dt, cs = crossfit(m, splits, fk)
        # per block deltas
        y, p = data[m]["yH"], data[m]["pH"]
        Ar = AH if m == "eui" else np.ones(len(H))
        blocks = []
        for (a, b), (k, c) in zip(splits, cs):
            pn = p[b] + c * xcol(m, p[b], Ar[b], MH[b], k, AB[m])
            blocks.append(round((nrmse(y[b], pn, m) - nrmse(y[b], p[b], m)) / 5 * 1e5, 3))
        det[m] = dict(dtotal=dt, block_dtotal_e5=blocks, k_c=cs)
        t += dt
    schemeC[f"k{'grid' if fk is None else fk}"] = dict(total=t, **det)
    print("C", f"k{'grid' if fk is None else fk}", round(t * 1e5, 3), {m: det[m]["block_dtotal_e5"] for m in det})
res["C_subn_blocks_H"] = schemeC

# ---- scheme D: reverse, fit on H corrected, score on P-side blend
schemeD = {}
t = 0.0
for m in ("gas", "eui"):
    d = data[m]
    ArP, ArH = (AP, AH) if m == "eui" else (np.ones(len(P)), np.ones(len(H)))
    k, c = fit_kc(m, d["pH"], d["yH"], ArH, MH, AB[m])
    xP = xcol(m, d["pP"], ArP, MP, k, AB[m])
    dt = (nrmse(d["yP"], d["pP"] + c * xP, m) - nrmse(d["yP"], d["pP"], m)) / 5
    schemeD[m] = dict(k_H=k, c_H=c, P_dtotal=dt)
    t += dt
schemeD["total"] = t
res["D_fitH_scoreP"] = schemeD
print("D fit H -> P:", {m: (schemeD[m]["k_H"], round(schemeD[m]["c_H"], 4), round(schemeD[m]["P_dtotal"] * 1e5, 3))
                        for m in ("gas", "eui")}, "total e-5", round(t * 1e5, 3))

# ---------------------------------------------------------------- 4. bootstrap + decomposition + hard clip (scheme A)
rng = np.random.default_rng(12345)
newH = {}
for m in ("gas", "eui"):
    d = data[m]
    Ar = AH if m == "eui" else 1.0
    newH[m] = d["pH"] + schemeA[m]["c_P"] * xcol(m, d["pH"], Ar, MH, schemeA[m]["k_P"], AB[m])
n = len(H)
bs = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    t = 0.0
    for m in ("gas", "eui"):
        y, p0, p1 = data[m]["yH"][i], data[m]["pH"][i], newH[m][i]
        t += (nrmse(y, p1, m) - nrmse(y, p0, m)) / 5
    bs.append(t)
bs = np.array(bs)
res["A_bootstrap"] = dict(mean=float(bs.mean()), sd=float(bs.std()), p_worse=float(np.mean(bs >= 0)))
print("A bootstrap total e-5: mean", round(bs.mean() * 1e5, 3), "sd", round(bs.std() * 1e5, 3), "P(>=0)", np.mean(bs >= 0))

# decomposition of the SSE gain by prediction band (gas scale), in total-score units
dec = {}
for m in ("gas", "eui"):
    d = data[m]
    y, p0, p1 = d["yH"], d["pH"], newH[m]
    pg = p0 * AH if m == "eui" else p0
    base = nrmse(y, p0, m)
    sse0 = float(np.sum((y - p0) ** 2))
    tab = {}
    for lo, hi in ((0, 7e5), (7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, np.inf)):
        g = (pg >= lo) & (pg < hi)
        dsse = float(np.sum((y[g] - p1[g]) ** 2) - np.sum((y[g] - p0[g]) ** 2))
        # first-order: dNRMSE = base * dsse / (2 sse0)
        tab[f"{lo:.0f}-{hi:.0f}"] = dict(n=int(g.sum()), n_moved_gt_1pct=int((np.abs(p1[g] - p0[g]) >
                                                                              0.01 * np.abs(p0[g])).sum()),
                                         dtotal_e5=round(base * dsse / (2 * sse0) / 5 * 1e5, 4))
    dec[m] = tab
    print("decomp", m, {b: (v["n"], v["n_moved_gt_1pct"], v["dtotal_e5"]) for b, v in tab.items()})
res["A_decomposition"] = dec

# hard clips for reference (no fitted parameter): gas <= U, EUI <= U / A
clip = {}
t = 0.0
for m in ("gas", "eui"):
    d = data[m]
    cap = U / AH if m == "eui" else U
    pn = np.minimum(d["pH"], cap)
    dt = (nrmse(d["yH"], pn, m) - nrmse(d["yH"], d["pH"], m)) / 5
    clip[m] = dict(rows=int((d["pH"] > cap).sum()), dtotal=dt)
    t += dt
clip["total"] = t
res["hard_clip_1e6"] = clip
print("hard clip at 1e6 total e-5", round(t * 1e5, 3), {m: (clip[m]["rows"], round(clip[m]["dtotal"] * 1e5, 3))
                                                         for m in ("gas", "eui")})
json.dump(res, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
