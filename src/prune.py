"""Greedy forward selection of blend members (in-sample NNLS via Gram matrices; honest CV at the end).

usage: python src/prune.py [max_models_per_family] [--exclude prefix1,...] [--include prefix1,...] [--hold]
--hold selects on the fold-averaged holdout predictions of exp_hold.py (rows H) instead of on OOF predictions.
"""
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(__file__))
from blend import cv_blend_score  # noqa: E402
from common import SIGMAS, TARGET_COLS, holdout_split, load, nrmse  # noqa: E402

ELEC = ("epm", "elec", "lepm", "epa", "lepa")


def gram_nnls(G, b):
    """argmin_{w>=0} w'Gw - 2b'w via a Cholesky factor (G = R'R)."""
    R = np.linalg.cholesky(G + 1e-9 * np.trace(G) / len(G) * np.eye(len(G))).T
    w, _ = nnls(R, np.linalg.solve(R.T, b))
    return w


def greedy(stats, n_rows, kmax):
    """stats: list of (G, b, yy, sigma) per metric sharing the same columns; minimises the sum of NRMSEs."""
    m = len(stats[0][1])
    chosen, path = [], []
    for _ in range(min(kmax, m)):
        best = (np.inf, None)
        for j in range(m):
            if j in chosen:
                continue
            S = chosen + [j]
            tot = 0.0
            for G, b, yy, sig in stats:
                w = gram_nnls(G[np.ix_(S, S)], b[S])
                sse = yy - 2 * b[S] @ w + w @ G[np.ix_(S, S)] @ w
                tot += np.sqrt(max(sse, 0) / n_rows) / sig
            if tot < best[0]:
                best = (tot, j)
        chosen.append(best[1])
        path.append(best[0])
    return chosen, path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kmax", type=int, nargs="?", default=25)
    ap.add_argument("--exclude", default="", help="comma-separated run-name prefixes to leave out")
    ap.add_argument("--include", default="", help="comma-separated run-name prefixes to keep (default: all)")
    ap.add_argument("--hold", action="store_true", help="select on holdout predictions (exp/hold/pred, rows H)")
    a = ap.parse_args()
    kmax, excl = a.kmax, tuple(p for p in a.exclude.split(",") if p)
    tr, _ = load()
    rows = holdout_split(len(tr))[1] if a.hold else np.arange(len(tr))
    src = "exp/hold/pred" if a.hold else "exp/oof"
    A = tr["bld_dim_06"].values[rows]
    y = {c: tr[c].values[rows] for c in TARGET_COLS}
    runs = sorted(f[:-4] for f in os.listdir(src)
                  if os.path.getsize(f"{src}/{f}") > 0 and os.path.exists(f"exp/pred/{f}"))
    runs = [r for r in runs if not (excl and r.startswith(excl))]
    incl = tuple(p for p in a.include.split(",") if p)
    runs = [r for r in runs if not incl or r.startswith(incl)]
    el = [r for r in runs if any(t in r.split("_") for t in ELEC) and not r.endswith("_ad")]
    gas = [r for r in runs if r not in el and not r.endswith("_ad")]
    n = len(A)

    Pg = np.column_stack([np.load(f"{src}/{r}.npy") for r in gas])
    ye = y[TARGET_COLS[2]]
    PgA = Pg * A[:, None]
    st_g = [(Pg.T @ Pg, Pg.T @ ye, ye @ ye, SIGMAS[TARGET_COLS[2]]),
            (PgA.T @ PgA, PgA.T @ (ye * A), (ye * A) @ (ye * A), SIGMAS[TARGET_COLS[0]])]
    cg, pg = greedy(st_g, n, kmax)
    Pe = np.column_stack([np.load(f"{src}/{r}.npy") for r in el])
    yl = y[TARGET_COLS[1]]
    st_e = [(Pe.T @ Pe, Pe.T @ yl, yl @ yl, SIGMAS[TARGET_COLS[1]])]
    ce, pe = greedy(st_e, n, kmax)

    print("gas family (in-sample eui+gas NRMSE):")
    for k, (j, s) in enumerate(zip(cg, pg), 1):
        print(f"  {k:2d} +{gas[j]:22s} {s:.5f}")
    print("elec family (in-sample elec NRMSE):")
    for k, (j, s) in enumerate(zip(ce, pe), 1):
        print(f"  {k:2d} +{el[j]:22s} {s:.5f}")
    print("gas runs:", ",".join(gas[j] for j in cg))
    print("elec runs:", ",".join(el[j] for j in ce))

    for k in (8, 12, 16, 20, kmax):
        g_idx, e_idx = cg[:k], ce[:k]
        s_gas = nrmse(y[TARGET_COLS[0]], cv_blend_score(Pg[:, g_idx], ye, A) * A, TARGET_COLS[0])
        s_eui = nrmse(ye, cv_blend_score(Pg[:, g_idx], ye), TARGET_COLS[2])
        s_el = nrmse(yl, cv_blend_score(Pe[:, e_idx], yl), TARGET_COLS[1])
        print(f"top-{k}: honest gas {s_gas:.5f} elec {s_el:.5f} eui {s_eui:.5f} "
              f"total {(s_gas + s_el + s_eui + 0.03565) / 5:.5f}", flush=True)


if __name__ == "__main__":
    main()
