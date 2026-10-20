"""verify2_postproc/indep r2: v19vce varying-coefficient blend (A2xM2xp2, members of exp/v19vce/members.yaml) on the
full-train member OOF, recomputed with an own memory-light NNLS (Gram matrix + Cholesky reduction), then
  (a) least-squares multiplicative small-M deltas (M 2-3, M 4) as used for the test candidate,
  (b) honest effect of S1 / S3 / S4 on the full OOF by cross-fitting the deltas / k / bounds over row folds
      (5-fold, seeds 0..19): a third, much larger evaluation sample (429k rows, 877 small-M rows) beside H and P.
The full-train OOF is uncorrected (no residual correction), so this checks the teacher-level effect.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/indep/r2_fulloof.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402  (production basis, identical to final/src/blend.py)
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
OUT = "exp/audit/verify2_postproc/indep"
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EL, EUI])
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
n = len(tr)
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
lam = json.load(open("exp/v19vce/resid_knn.json"))
print("vc spec in resid_knn.json:", lam.get("vc"), flush=True)


def solve_nnls(Xfn, ncol, yv, s, chunk=25000):
    """argmin_{w>=0} ||s (y - X w)||^2 through the normal equations: G = X'S^2X = LL', then nnls(L', L^-1 X'S^2y)"""
    G = np.zeros((ncol, ncol))
    g = np.zeros(ncol)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        Xc = Xfn(sl) * s[sl, None]
        G += Xc.T @ Xc
        g += Xc.T @ (yv[sl] * s[sl])
    L = np.linalg.cholesky(G + 1e-12 * np.trace(G) / ncol * np.eye(ncol))
    w, _ = nnls(L.T, np.linalg.solve(L, g), maxiter=100 * ncol)
    return w


full = {}
for fam, names in (("gas", mem["gas"]), ("elec", mem["elec"])):
    X = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in names]).astype(float)
    for met in (("eui", "gas") if fam == "gas" else ("elec",)):
        yv = tr[EL].values if met == "elec" else tr[EUI].values
        s = A if met == "gas" else np.ones(n)
        w0 = solve_nnls(lambda sl: X[sl], X.shape[1], yv, s)
        vf = dict(A=A, M=M, p=X @ w0)
        B = vc_basis(lam["vc"][met], vf, vf)
        k = B.shape[1]
        W = solve_nnls(lambda sl: np.concatenate([X[sl] * B[sl, [j]] for j in range(k)], axis=1), k * X.shape[1], yv, s)
        W = W.reshape(k, X.shape[1]).T
        p = ((X @ W) * B).sum(axis=1)
        if met == "gas":
            full[met] = (tr[GAS].values.astype(float), p * A)
        else:
            full[met] = (yv.astype(float), p)
        yy, pp = full[met]
        print(met, "members", len(names), "basis", k, "full-OOF rmse", float(np.sqrt(np.mean((yy - pp) ** 2))),
              "nrmse", float(np.sqrt(np.mean((yy - pp) ** 2)) / SIG[met]), flush=True)
    del X
MET = ["gas", "eui", "elec"]
GROUPS = [(2, 3), (4, 4)]
res = {"nrmse_full_oof": {m: float(np.sqrt(np.mean((full[m][0] - full[m][1]) ** 2)) / SIG[m]) for m in MET}}


def deltas(p, r, Mv):
    return [float(p[(Mv >= lo) & (Mv <= hi)] @ r[(Mv >= lo) & (Mv <= hi)] /
                  (p[(Mv >= lo) & (Mv <= hi)] @ p[(Mv >= lo) & (Mv <= hi)])) for lo, hi in GROUPS]


res["deltas_full_oof"] = {m: deltas(full[m][1], full[m][0] - full[m][1], M) for m in MET}
res["counts"] = {"M<=3": int((M <= 3).sum()), "M=4": int((M == 4).sum()), "M_min": float(M.min())}
print("full-OOF LS deltas [M2-3, M4]:", res["deltas_full_oof"], res["counts"], flush=True)
yG = full["gas"][0]
res["bounds_train"] = dict(gmax=float(yG.max()), euimin=float(tr[EUI].min()), eamin=float((tr[EL] / A).min()),
                           eamax=float((tr[EL] / A).max()))
np.savez_compressed(f"{OUT}/r2_fulloof_pred.npz", **{f"{m}_p": full[m][1] for m in MET}, **{f"{m}_y": full[m][0] for m in MET})


def tn_shift(p, s, U):
    b = (U - p) / s
    return -s * np.exp(norm.logpdf(b) - log_ndtr(b))


def nr(m, idx, p):
    return float(np.sqrt(np.mean((full[m][0][idx] - p[idx]) ** 2)) / SIG[m])


# cross-fitted chain on the full OOF: deltas, variance model, k and bounds all from the fit folds only
allr = np.arange(n)
summ = {k: [] for k in ("S1", "S3", "S4", "S1+S3+S4")}
ks = []
for seed in range(20):
    pred = {st: {m: np.empty(n) for m in MET} for st in ("S0", "S1", "S1S3", "S1S3S4", "S3only", "S4only")}
    for a, b in KFold(5, shuffle=True, random_state=seed).split(allr):
        ya = {m: full[m][0][a] for m in MET}
        bd = dict(gmax=ya["gas"].max(), euimin=ya["eui"].min(), eamin=(ya["elec"] / A[a]).min(), eamax=(ya["elec"] / A[a]).max())
        for m in MET:
            p = full[m][1]
            pred["S0"][m][b] = p[b]
            ds = deltas(p[a], ya[m] - p[a], M[a])
            q = p.copy()
            for (lo, hi), d in zip(GROUPS, ds):
                q[(M >= lo) & (M <= hi)] *= 1 + d
            pred["S1"][m][b] = q[b]
            lo_b, hi_b = {"gas": (bd["euimin"] * A, np.full(n, bd["gmax"])), "eui": (np.full(n, bd["euimin"]), bd["gmax"] / A),
                          "elec": (bd["eamin"] * A, bd["eamax"] * A)}[m]
            outs = {}
            for lab, start in (("S1S3", q), ("S3only", p)):
                if m in ("gas", "eui"):
                    vm = np.linalg.lstsq(np.column_stack([np.ones(len(a)), 1 / M[a]]), ((ya[m] - p[a]) / p[a]) ** 2, rcond=None)[0]
                    sd = start * np.sqrt(np.maximum(vm[0] + vm[1] / M, 1e-8))
                    x = tn_shift(start, sd, bd["gmax"] if m == "gas" else bd["gmax"] / A)
                    k = float(x[a] @ (ya[m] - start[a]) / (x[a] @ x[a]))
                    if lab == "S1S3":
                        ks.append((m, k))
                    outs[lab] = start + k * x
                else:
                    outs[lab] = start
            pred["S1S3"][m][b] = outs["S1S3"][b]
            pred["S1S3S4"][m][b] = np.clip(outs["S1S3"], lo_b, hi_b)[b]
            pred["S3only"][m][b] = outs["S3only"][b]
            pred["S4only"][m][b] = np.clip(p, lo_b, hi_b)[b]
    for lab, st in (("S1", "S1"), ("S3", "S3only"), ("S4", "S4only"), ("S1+S3+S4", "S1S3S4")):
        per = {m: nr(m, allr, pred[st][m]) - nr(m, allr, pred["S0"][m]) for m in MET}
        summ[lab].append(per)
    print("seed", seed, {lab: round(sum(v[-1].values()) / 5, 8) for lab, v in summ.items()}, flush=True)
res["crossfit_full_oof"] = {lab: dict(per_mean={m: float(np.mean([p[m] for p in v])) for m in MET},
                                      total_mean=float(np.mean([sum(p.values()) / 5 for p in v])),
                                      total_sd_seeds=float(np.std([sum(p.values()) / 5 for p in v]))) for lab, v in summ.items()}
res["k_full_oof"] = {m: [float(np.mean([k for mm, k in ks if mm == m])), float(np.std([k for mm, k in ks if mm == m]))]
                     for m in ("gas", "eui")}
print(json.dumps(res["crossfit_full_oof"], indent=1), res["k_full_oof"], flush=True)
json.dump(res, open(f"{OUT}/r2_fulloof.json", "w"), indent=1)
print("wrote", f"{OUT}/r2_fulloof.json")
