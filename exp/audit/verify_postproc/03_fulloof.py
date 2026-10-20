"""Verify postproc S1, step 3: independent refit of the v19vce VC blend (A2xM2xp2) on the full-train member OOF,
validated against the pipeline's own test base prediction (exp/pl/teacher_deploy.npz), then a robustness study of the
small-M bias on the full-train OOF residuals:

- per M group: LS delta sum(p r)/sum(p^2), heteroscedasticity-robust and jackknife se, median / trimmed relative
  residual, share of rows with r > 0, share of sum(p r) from the top 5 rows, LS delta without the top 1/3/5 rows;
- bootstrap simulation of the REALIZED test change (joint over metrics): deltas re-estimated on a bootstrap of the
  136 / 741 train rows, applied to independently drawn test-sized groups (60 / 298 rows), scored on the full test and on
  a random 70% (private) / 30% (public) split of those rows.

NNLS is solved through the Gram matrix (accumulated in row chunks) with scipy lsq_linear(method='bvls') on the
Cholesky factor (a different solver from the pipeline's scipy.optimize.nnls and from postproc/12's).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/03_fulloof.py
"""
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import lsq_linear

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
COL = {"gas": GAS, "eui": EUI, "elec": EL}
SIG = {m: SIGMAS[c] for m, c in COL.items()}
MET = ("gas", "eui", "elec")
SPEC = "A2xM2xp2"

tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
td = np.load("exp/pl/teacher_deploy.npz", allow_pickle=False)


def nnls_gram(G, b):
    L = np.linalg.cholesky(G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G)))
    res = lsq_linear(L.T, np.linalg.solve(L, b), bounds=(0, np.inf), method="bvls", tol=1e-14, max_iter=10000)
    return res.x


def gram(Xfn, n, y, s, chunk=16384):
    G, b = 0.0, 0.0
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        Xc = Xfn(sl) * s[sl, None]
        G = G + Xc.T @ Xc
        b = b + Xc.T @ (y[sl] * s[sl])
    return G, b


fit, app = {}, {}
for fam, names in (("gas", mem["gas"]), ("elec", mem["elec"])):
    Pf = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
    Pa = np.column_stack([np.load(f"exp/pred/{n}.npy") for n in names])
    for met in (("eui", "gas") if fam == "gas" else ("elec",)):
        y = tr[EL].values if met == "elec" else tr[EUI].values
        s = A if met == "gas" else np.ones(len(y))
        w0 = nnls_gram(*gram(lambda sl: Pf[sl], len(y), y, s))
        vf, va = dict(A=A, M=M, p=Pf @ w0), dict(A=At, M=Mt, p=Pa @ w0)
        Bf, Ba = vc_basis(SPEC, vf, vf), vc_basis(SPEC, vf, va)
        k = Pf.shape[1]
        W = nnls_gram(*gram(lambda sl: np.concatenate([Pf[sl] * Bf[sl, [j]] for j in range(Bf.shape[1])], axis=1),
                            len(y), y, s)).reshape(Bf.shape[1], k).T
        pf, pa = ((Pf @ W) * Bf).sum(1), ((Pa @ W) * Ba).sum(1)
        if met == "gas":
            pf, pa, yy = pf * A, pa * At, y * A
        else:
            yy = y
        rel = np.abs(pa - td[met]) / np.maximum(np.abs(td[met]), 1e-9)
        print(f"{met}: refit vs pipeline test base (teacher_deploy): max rel diff {rel.max():.2e}, rms rel {np.sqrt(np.mean(rel ** 2)):.2e}; "
              f"in-sample NRMSE {np.sqrt(np.mean((yy - pf) ** 2)) / SIG[met]:.5f}", flush=True)
        fit[met], app[met] = (yy, pf), pa
    del Pf, Pa

groups = {"M2-3": (M >= 2) & (M <= 3), "M4": M == 4, "M5": M == 5, "M6-8": (M >= 6) & (M <= 8), "M>=9": M >= 9}
print("\nrobustness of the bias on the full-train OOF residuals (r = y - p, metric scale)")
for met in MET:
    y, p = fit[met]
    r = y - p
    print(f"== {met}")
    for g, msk in groups.items():
        pg, rg = p[msk], r[msk]
        n = len(pg)
        S = pg @ pg
        d = float(pg @ rg / S)
        e = rg - d * pg
        se_hc0 = float(np.sqrt(np.sum(pg ** 2 * e ** 2)) / S)
        if n <= 20000:
            dl = (pg @ rg - pg * rg) / (S - pg ** 2)
            se_jk = float(np.sqrt((n - 1) / n * np.sum((dl - dl.mean()) ** 2)))
        else:
            se_jk = float("nan")
        relg = rg / pg
        q = np.sort(relg)
        tm = q[int(0.1 * n): n - int(0.1 * n)].mean()
        contrib = pg * rg
        order = np.argsort(-contrib)
        top5 = contrib[order[:5]].sum() / contrib.sum()
        wo = []
        for kk in (1, 3, 5):
            keep = np.ones(n, bool)
            keep[order[:kk]] = False
            wo.append(round(float(pg[keep] @ rg[keep] / (pg[keep] @ pg[keep])), 4))
        print(f"   {g:5s} n {n:6d} LS d {d:+.4f} (se HC0 {se_hc0:.4f}, jackknife {se_jk:.4f}) | rel median {np.median(relg):+.4f} "
              f"trim10 {tm:+.4f} share r>0 {np.mean(rg > 0):.3f} | top5 share of sum(pr) {top5:.2f}; d w/o top1/3/5 {wo}")

# ---- bootstrap simulation of the realized test change
rng = np.random.default_rng(2026)
nrm = {met: float(np.sqrt(np.mean((fit[met][0] - fit[met][1]) ** 2)) / SIG[met]) for met in MET}
idx_g = {"M2-3": np.where(groups["M2-3"])[0], "M4": np.where(groups["M4"])[0]}
n_test = {"M2-3": int(((Mt >= 2) & (Mt <= 3)).sum()), "M4": int((Mt == 4).sum())}
NT = len(Mt)
B = 20000
out = {k: np.zeros(B) for k in ("full", "private", "public")}
for b in range(B):
    for g, ix in idx_g.items():
        est = rng.choice(ix, len(ix), replace=True)
        tst = rng.choice(ix, n_test[g], replace=True)
        priv = rng.random(n_test[g]) < 0.7
        for met in MET:
            y, p = fit[met]
            d = float(p[est] @ (y[est] - p[est]) / (p[est] @ p[est]))
            pt, rt = p[tst], y[tst] - p[tst]
            dsse = d * d * pt ** 2 - 2 * d * pt * rt
            scale = 2 * SIG[met] ** 2 * nrm[met] * 5
            out["full"][b] += dsse.sum() / NT / scale
            out["private"][b] += dsse[priv].sum() / (0.7 * NT) / scale
            out["public"][b] += dsse[~priv].sum() / (0.3 * NT) / scale
print(f"\nrealized test change, bootstrap B={B} (deltas re-estimated each draw; test groups of {n_test})")
for k, v in out.items():
    print(f"   {k:8s} mean {v.mean():+.8f} sd {v.std():.8f} P(<0) {np.mean(v < 0):.3f} q05/50/95 {np.round(np.quantile(v, [0.05, 0.5, 0.95]), 8).tolist()}")
