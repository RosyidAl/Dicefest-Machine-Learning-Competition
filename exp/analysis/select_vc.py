"""Member selection under the varying-coefficient blend (D012): start from the v19 blend, greedy forward addition with
a fast screen, then a backward pass. Score = honest 5-fold nested VC blend (gas + EUI for the gas family, elec for the
elec family), same as the v19 selection but with blend.blend_vc instead of one global NNLS weight vector.

Screen: share of the current nested residual explained (least squares, all rows) by a candidate's spec-expanded
column block; only the TOPK best-screened candidates are scored exactly.

usage: python exp/analysis/select_vc.py <spec> <out.json> [gas|elec]   (env CANDS=a,b: only these candidates)
"""
import json
import os
import sys

import numpy as np
import yaml
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from blend import disagreement, predict_vc, vc_basis  # noqa: E402
from common import TARGET_COLS, load, nrmse  # noqa: E402

SPEC, OUT = sys.argv[1], sys.argv[2]
FAMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gas", "elec"]
TOPK, EPS = 4, 2e-6
ELEC = ("epm", "elec", "lepm", "epa", "lepa")
SKIP = ("c4_", "cat_")  # CatBoost GPU is not deterministic (D004)
is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
tr, _ = load()
A = tr.bld_dim_06.values
V = dict(A=A, M=tr.bld_dim_12.values.astype(float))
yg, ye, yl = (tr[c].values for c in (TARGET_COLS[0], TARGET_COLS[2], TARGET_COLS[1]))
cfg = yaml.safe_load(open("exp/v19cfg/members.yaml"))
names = sorted(f[:-4] for f in os.listdir("exp/oof") if f.endswith(".npy") and not f.startswith(SKIP))
COL = {}


def col(r):
    if r not in COL:
        COL[r] = np.load(f"exp/oof/{r}.npy")
    return COL[r]


FOLDS = list(KFold(5, shuffle=True, random_state=0).split(A))


def nnls_blend(X, y, s=None):
    """NNLS through the normal equations + Cholesky (same minimiser as blend.nnls_blend up to rounding; ~15x faster)"""
    if s is not None:
        X, y = X * s[:, None], y * s
    G, bv = X.T @ X, X.T @ y  # normal equations: ||Xw - y||^2 = ||Rw - c||^2 + const with G = R'R, R'c = b
    G[np.diag_indices_from(G)] += 1e-12 * np.trace(G) / len(G)
    L = np.linalg.cholesky(G)
    return nnls(L.T, solve_triangular(L, bv, lower=True), maxiter=50 * X.shape[1])[0]


def nnls_blend_vc(X, y, B, s=None):
    Z = np.concatenate([X * B[:, [j]] for j in range(B.shape[1])], axis=1)
    return nnls_blend(Z, y, s).reshape(B.shape[1], X.shape[1]).T


def honest(X, y, s):
    pred = np.zeros(len(y))
    for a, b in FOLDS:
        w = nnls_blend(X[a], y[a], s[a])
        va = {k: v[a] for k, v in V.items()}
        va["p"], va["d"] = X[a] @ w, disagreement(X[a])
        vb = {k: v[b] for k, v in V.items()}
        vb["p"], vb["d"] = X[b] @ w, disagreement(X[b])
        W = nnls_blend_vc(X[a], y[a], vc_basis(SPEC, va, va), s[a])
        pred[b] = predict_vc(X[b], W, vc_basis(SPEC, va, vb))
    return pred


def fam_eval(sel, fam):
    """honest score and the nested residuals per metric (residuals on the blend's fit scale)"""
    X = np.column_stack([col(r) for r in sel])
    if fam == "gas":
        pe = honest(X, ye, np.ones(len(A)))
        pg = honest(X, ye, A)
        sc = nrmse(yg, pg * A, TARGET_COLS[0]) + nrmse(ye, pe, TARGET_COLS[2])
        return sc, [(ye - pg) * A, ye - pe], X
    pl = honest(X, yl, np.ones(len(A)))
    return nrmse(yl, pl, TARGET_COLS[1]), [yl - pl], X


def screen(X, res, cands, fam):
    """explained share of each residual by the candidate's expanded block (full-data basis with the global p)"""
    out = []
    bases = []
    for i, r in enumerate(res):
        s = A if (fam == "gas" and i == 0) else np.ones(len(A))
        w = nnls_blend(X, ye if fam == "gas" else yl, s)
        vv = dict(V, p=X @ w, d=disagreement(X))
        B = vc_basis(SPEC, vv, vv)
        bases.append((B, s))
    for c in cands:
        tot = 0.0
        for (B, s), r in zip(bases, res):
            Z = col(c)[:, None] * B * s[:, None]
            coef = np.linalg.lstsq(Z, r, rcond=None)[0]
            tot += 1 - np.sum((r - Z @ coef) ** 2) / np.sum(r ** 2)
        out.append((tot, c))
    return sorted(out, reverse=True)


result = {}
for fam in FAMS:
    sel = list(cfg["blend"][fam])
    cur, res, X = fam_eval(sel, fam)
    print(f"[{fam}] v19 members under {SPEC}: {cur:.6f}", flush=True)
    pool = [n for n in names if n not in sel and is_el(n) == (fam == "elec")]
    if os.environ.get("CANDS"):  # explicit candidate list (e.g. new members): every one is scored exactly
        pool = [n for n in os.environ["CANDS"].split(",") if n in pool]
    while pool:
        top = pool if os.environ.get("CANDS") else [c for _, c in screen(X, res, pool, fam)[:TOPK]]
        scored = []
        for c in top:
            sc, _, _ = fam_eval(sel + [c], fam)
            scored.append((sc, c))
            print(f"   try +{c}: {sc:.6f}", flush=True)
        best = min(scored)
        if best[0] < cur - EPS:
            sel.append(best[1])
            pool.remove(best[1])
            cur, res, X = fam_eval(sel, fam)
            print(f"+{best[1]} {fam} {cur:.6f}", flush=True)
        else:
            break
    if os.environ.get("BACKWARD"):
        for r in list(sel):  # backward
            sc, _, _ = fam_eval([x for x in sel if x != r], fam)
            if sc <= cur + 1e-7:
                sel.remove(r)
                cur = sc
                print(f"-{r} {fam} {sc:.6f}", flush=True)
    result[fam] = {"members": sel, "score": cur}
    json.dump(result, open(OUT, "w"), indent=1)
print("done", {f: (len(v["members"]), round(v["score"], 6)) for f, v in result.items()})
