"""Full-train OOF re-check of the small-meter bias, with my own VC-blend refit (TSQR + NNLS, same spec A2xM2xp2).

1. Rebuild the v19vce varying-coefficient blend in-sample on the member OOF arrays (exp/oof) for gas / EUI / elec.
2. Per-M bias with sandwich standard errors and robustness checks (trimmed, drop top rows, sign fraction).
3. Honest K-fold cross-fitting of the per-group multiplicative shift over all train rows (many seeds).
Outputs: v02_fulloof.json, v02_pf.npz (fitted predictions on the metric scale, for later steps).
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

sys.path.insert(0, "final/src")
from blend import disagreement, nnls_blend, predict_vc, vc_basis  # noqa: E402

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
SPEC = "A2xM2xp2"
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EL, EUI])
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
y = {"gas": tr[GAS].values.astype(float), "eui": tr[EUI].values.astype(float), "elec": tr[EL].values.astype(float)}
del tr
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
PF = {fam: np.column_stack([np.load(f"exp/oof/{r}.npy").astype(float) for r in mem[fam]]) for fam in ("gas", "elec")}
n = len(A)
out = {"n": n}


def tsqr_nnls(P, B, yv, s, chunk=40000):
    """NNLS on X = [P*B_0 | P*B_1 | ...] with row scale s via a chunked QR of [s*X | s*y] (exact same minimiser)."""
    k = P.shape[1] * B.shape[1]
    R = None
    for a in range(0, len(yv), chunk):
        b = slice(a, a + chunk)
        Z = np.concatenate([P[b] * B[b, [j]] for j in range(B.shape[1])] + [yv[b, None]], axis=1) * s[b, None]
        R = np.linalg.qr(Z if R is None else np.vstack([R, Z]), mode="r")
    w, _ = nnls(R[:k, :k], R[:k, k], maxiter=50 * k)
    return w.reshape(B.shape[1], P.shape[1]).T


pf = {}
for met in ("gas", "eui", "elec"):
    fam = "elec" if met == "elec" else "gas"
    P = PF[fam]
    yv = y["eui"] if met in ("gas", "eui") else y["elec"]
    s = A if met == "gas" else np.ones(n)
    wg = nnls_blend(P, yv, s if met == "gas" else None)
    vf = dict(A=A, M=M, p=P @ wg, d=disagreement(P))
    Bf = vc_basis(SPEC, vf, vf)
    W = tsqr_nnls(P, Bf, yv, s)
    p = predict_vc(P, W, Bf)
    pf[met] = p * A if met == "gas" else p
    pg = (P @ wg) * (A if met == "gas" else 1.0)
    out[f"{met}_insample_nrmse_vc"] = float(np.sqrt(np.mean((y[met] - pf[met]) ** 2)) / SIG[met])
    out[f"{met}_insample_nrmse_global"] = float(np.sqrt(np.mean((y[met] - pg) ** 2)) / SIG[met])
    print(f"{met}: in-sample NRMSE VC {out[f'{met}_insample_nrmse_vc']:.5f} global {out[f'{met}_insample_nrmse_global']:.5f}",
          flush=True)
np.savez("exp/audit/verify2_postproc/v02_pf.npz", **{m: pf[m] for m in pf}, M=M, A=A)
del PF


def ls_delta(p, r):
    d = float(p @ r / (p @ p))
    se = float(np.sqrt(np.sum((p ** 2) * (r - d * p) ** 2)) / (p @ p))  # sandwich (heteroscedasticity-robust)
    return d, se


GROUPS = {"G23": (M >= 2) & (M <= 3), "G4": M == 4}
bias = {}
for met in pf:
    p, r = pf[met], y[met] - pf[met]
    rows = {}
    for name, g in list(GROUPS.items()) + [(f"M={v}", M == v) for v in (5, 6, 7, 8)] + [("M>=9", M >= 9)]:
        d, se = ls_delta(p[g], r[g])
        rel = r[g] / p[g]
        o = np.argsort(-np.abs(r[g]))  # drop the 5 largest absolute residuals
        keep = np.ones(g.sum(), bool)
        keep[o[:5]] = False
        d5, _ = ls_delta(p[g][keep], r[g][keep])
        lo, hi = np.quantile(rel, [0.05, 0.95])
        tm = rel[(rel >= lo) & (rel <= hi)].mean()
        rows[name] = dict(n=int(g.sum()), ls_delta=d, se=se, t=d / se, mean_rel=float(rel.mean()),
                          trimmed_mean_rel=float(tm), median_rel=float(np.median(rel)),
                          frac_pos=float((r[g] > 0).mean()), ls_delta_drop5=d5)
    bias[met] = rows
    print(f"== {met}")
    for k_, v in rows.items():
        print(f"   {k_:5s} n {v['n']:6d} ls {v['ls_delta']:+.5f} (se {v['se']:.4f}, t {v['t']:+.1f}) mean_rel {v['mean_rel']:+.4f} "
              f"trim {v['trimmed_mean_rel']:+.4f} med {v['median_rel']:+.4f} frac_pos {v['frac_pos']:.3f} drop5 {v['ls_delta_drop5']:+.4f}")
out["bias"] = bias


GIDX = {g: np.where(mask)[0] for g, mask in GROUPS.items()}


def cross_fit(p, yv, sig, K, seed, groups=("G23", "G4")):
    """random K-fold over all rows (only the group rows' fold labels matter): shift fitted on the other folds' group
    rows, applied to this fold's group rows. Exact NRMSE change from the SSE difference."""
    rng = np.random.default_rng(seed)
    sse0 = np.sum((yv - p) ** 2)
    sse1 = sse0
    for gname in groups:
        ix = GIDX[gname]
        fold = rng.integers(0, K, len(ix))
        pg, yg = p[ix], yv[ix]
        q = pg.copy()
        for k in range(K):
            trn = fold != k
            d, _ = ls_delta(pg[trn], yg[trn] - pg[trn])
            q[~trn] = pg[~trn] * (1 + d)
        q = np.clip(q, 0, None)
        sse1 = sse1 - np.sum((yg - pg) ** 2) + np.sum((yg - q) ** 2)
    return (np.sqrt(sse1 / len(p)) - np.sqrt(sse0 / len(p))) / sig


cf = {}
for K in (2, 5, 10):
    for groups in (("G23", "G4"), ("G23",), ("G4",)):
        tots, per = [], {m: [] for m in pf}
        for seed in range(200):
            t = 0.0
            for met in pf:
                d = cross_fit(pf[met], y[met], SIG[met], K, 5000 + seed, groups)
                per[met].append(d)
                t += d / 5
            tots.append(t)
        tots = np.array(tots)
        key = f"K{K}_{'+'.join(groups)}"
        cf[key] = dict(mean_total=float(tots.mean()), sd_seeds=float(tots.std()), frac_neg=float((tots < 0).mean()),
                       per_metric={m: float(np.mean(per[m])) for m in pf})
        print(f"cross-fit {key:10s}: total {tots.mean():+.8f} (sd seeds {tots.std():.8f}, frac<0 {(tots < 0).mean():.3f}) | "
              + " ".join(f"{m} {np.mean(per[m]):+.7f}" for m in pf), flush=True)
out["cross_fit"] = cf

# in-sample (optimistic) gain with the all-row deltas, for reference
ins, deltas = 0.0, {}
for met in pf:
    p, yv = pf[met], y[met]
    deltas[met] = {gname: ls_delta(p[g], yv[g] - p[g])[0] for gname, g in GROUPS.items()}
    q = p.copy()
    for gname, g in GROUPS.items():
        q[g] = p[g] * (1 + deltas[met][gname])
    ins += (np.sqrt(np.mean((yv - q) ** 2)) - np.sqrt(np.mean((yv - p) ** 2))) / SIG[met] / 5
out["deltas_all_rows"] = deltas
out["insample_total_gain"] = float(ins)
print("deltas (all train rows):", json.dumps({m: {g: round(v, 5) for g, v in d.items()} for m, d in deltas.items()}))
print(f"in-sample total gain (optimistic) {ins:+.8f}")
json.dump(out, open("exp/audit/verify2_postproc/v02_fulloof.json", "w"), indent=1)
