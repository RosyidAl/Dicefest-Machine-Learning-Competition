"""Audit postproc 14: functional form of the small-M bias. Is it multiplicative (relative bias constant in p) or
additive? Uses (i) the full-train OOF of the v19vce VC blend (recomputed as in 12) and (ii) P-OOF teacher -> H
corrected honest transfer (chain_H.npz).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/14_smallM_form.py
"""
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", EL, EUI])
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
from scipy.optimize import nnls  # noqa: E402


def nnls_gram(G, b):
    ridge = 1e-12 * np.trace(G) / len(G)
    L = np.linalg.cholesky(G + ridge * np.eye(len(G)))
    return nnls(L.T, np.linalg.solve(L, b), maxiter=50 * len(G))[0]


def gram(X_fn, n, y, s, chunk=20000):
    G = b = None
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        Xc = X_fn(sl) * s[sl, None]
        yc = y[sl] * s[sl]
        G = Xc.T @ Xc if G is None else G + Xc.T @ Xc
        b = Xc.T @ yc if b is None else b + Xc.T @ yc
    return G, b


full = {}
for fam, names in (("gas", mem["gas"]), ("elec", mem["elec"])):
    X = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
    mets = {"eui": (tr[EUI].values, None), "gas": (tr[EUI].values, A)} if fam == "gas" else {"elec": (tr[EL].values, None)}
    for met, (y, rs) in mets.items():
        s = np.ones(len(y)) if rs is None else rs
        w0 = nnls_gram(*gram(lambda sl: X[sl], len(y), y, s))
        vf = dict(A=A, M=M, p=X @ w0)
        B = vc_basis("A2xM2xp2", vf, vf)
        W = nnls_gram(*gram(lambda sl: np.concatenate([X[sl] * B[sl, [j]] for j in range(B.shape[1])], axis=1),
                            len(y), y, s)).reshape(B.shape[1], X.shape[1]).T
        p = ((X @ W) * B).sum(axis=1)
        full[met] = (y * A, p * A) if met == "gas" else (y, p)
np.savez_compressed("exp/audit/postproc/14_fullvc_oof.npz", **{f"{m}_y": v[0] for m, v in full.items()},
                    **{f"{m}_p": v[1] for m, v in full.items()})

print("(i) full-train VC OOF: relative bias by p tercile within the group (n, sum r / sum p, mean r / sigma)")
for met, (y, p) in full.items():
    for gname, g in (("M<=3", (M >= 2) & (M <= 3)), ("M=4", M == 4)):
        idx = np.where(g)[0]
        e = np.quantile(p[idx], [0, 1 / 3, 2 / 3, 1])
        cells = []
        for i in range(3):
            c = idx[(p[idx] >= e[i]) & (p[idx] <= e[i + 1])]
            r = y[c] - p[c]
            cells.append(f"p~{p[c].mean():.4g}: n {len(c)} rel {r.sum() / p[c].sum():+.3f} add {r.mean() / SIG[met]:+.4f}")
        print(f"   {met:4s} {gname:5s} | " + " | ".join(cells))

print("(ii) honest P-OOF -> H: multiplicative vs additive vs both, groups M<=3 (2..3) and M=4")
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, Mall = z["P"], z["H"], z["M"]
MP, MH = Mall[P], Mall[H]
tot = {}
for met in ("gas", "eui", "elec"):
    pP, rP = z[f"{met}_pP"], z[f"{met}_rP"]
    yH, pH = z[f"{met}_y"], np.clip(z[f"{met}_corrected"], 0, None)
    base = np.sqrt(np.mean((yH - pH) ** 2)) / SIG[met]
    for form in ("mult", "add", "both"):
        q = pH.copy()
        for lo, hi in ((2, 3), (4, 4)):
            gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
            if form == "mult":
                q[gH] = pH[gH] * (1 + pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
            elif form == "add":
                q[gH] = pH[gH] + rP[gP].mean()
            else:
                Xg = np.column_stack([np.ones(gP.sum()), pP[gP]])
                c = np.linalg.lstsq(Xg, rP[gP], rcond=None)[0]
                q[gH] = pH[gH] + c[0] + c[1] * pH[gH]
        d = np.sqrt(np.mean((yH - q) ** 2)) / SIG[met] - base
        tot[form] = tot.get(form, 0) + d / 5
        print(f"   {met:4s} {form:5s} d {d:+.7f} total {d / 5:+.8f}")
print("TOTAL", {k: round(v, 8) for k, v in tot.items()})
