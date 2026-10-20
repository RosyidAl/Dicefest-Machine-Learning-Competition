"""Audit postproc 12 (copy for v19nca: members exp/v19nca): small-M bias of the actual v19nca blend (varying-coefficient NNLS A2xM2xp2, weights fitted on
the full-train member OOF as in final/src/pipeline._assemble) on the full-train OOF rows. Memory-light: the NNLS is
solved on the Gram matrix (Cholesky transform), accumulated in row chunks, instead of materialising the
429k x 216 design. Output: relative bias / least-squares multiplicative delta per M group, saved to json for the
test candidate (13_candidate.py).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/12_fullvc_smallM.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402
from common import TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", EL, EUI])
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
mem = yaml.safe_load(open("exp/v19nca2/members.yaml"))["blend"]
SPEC = "A2xM2xp2"


def nnls_gram(G, b):
    """argmin_{w>=0} w'Gw - 2b'w via Cholesky G = LL' -> nnls(L', L^-1 b)"""
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


out = {}
groups = {"M<=3": M <= 3, "M=4": M == 4, "M=5": M == 5, "M>=6": M >= 6}
for fam, names in (("gas", mem["gas"]), ("elec", mem["elec"])):
    X = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
    mets = {"eui": (tr[EUI].values, None), "gas": (tr[EUI].values, A)} if fam == "gas" else {"elec": (tr[EL].values, None)}
    for met, (y, rs) in mets.items():
        s = np.ones(len(y)) if rs is None else rs
        w0 = nnls_gram(*gram(lambda sl: X[sl], len(y), y, s))
        p0 = X @ w0
        vf = dict(A=A, M=M, p=p0)
        B = vc_basis(SPEC, vf, vf)
        G, bb = gram(lambda sl: np.concatenate([X[sl] * B[sl, [j]] for j in range(B.shape[1])], axis=1), len(y), y, s)
        W = nnls_gram(G, bb).reshape(B.shape[1], X.shape[1]).T
        p = ((X @ W) * B).sum(axis=1)
        yy, pp = (y * A, p * A) if met == "gas" else (y, p)
        r = yy - pp
        res = {}
        for gname, g in groups.items():
            rel = r[g] / pp[g]
            res[gname] = dict(n=int(g.sum()), mean_rel=float(rel.mean()), t=float(rel.mean() / (rel.std() / np.sqrt(g.sum()))),
                              ratio_of_sums=float(r[g].sum() / pp[g].sum()), ls_delta=float(pp[g] @ r[g] / (pp[g] @ pp[g])))
        out[met] = res
        print(met, "overall rmse", float(np.sqrt(np.mean(r ** 2))), {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()}, flush=True)
json.dump(out, open("exp/audit/postproc_nca2/12_fullvc_smallM_nca2.json", "w"), indent=1)
