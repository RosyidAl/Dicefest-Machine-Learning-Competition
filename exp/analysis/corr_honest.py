"""Honest check of PP0006 (correction-config reselection): the greedy selection on all of H is optimistic. Split H in
two halves; select configs on one half (same greedy + 5-fold cross-fitted lstsq as select_corr.py), score them on the
other half (cross-fitted there), and compare with the current CONFIGS scored the same way. Uses the cached candidate
columns of exp/corrsel/ (built by select_corr.py), so it only needs the residuals again.

usage: python exp/analysis/corr_honest.py [augment]   (augment: greedy additions to the current CONFIGS instead of a
       selection from scratch)
       python exp/analysis/corr_honest.py final eui[,gas]   augment selection on all of H for the listed metrics, the
       others keep CONFIGS -> exp/analysis/corr_final.json ({metric: {"configs": [...]}}, for src/chain_vc.py)
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
sys.path.insert(0, "exp/analysis")
from blend import blend_vc  # noqa: E402
from common import SIGMAS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402
from resid_corr import EL, EUI, METRICS, X8, features, group_keys  # noqa: E402
from resid_knn import CONFIGS  # noqa: E402
from select_corr import CACHE, EPS, GRID, SPEC, cv_rmse  # noqa: E402

AUG = len(sys.argv) > 1 and sys.argv[1] in ("augment", "final")
FINAL = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[1] == "final" else None


def greedy(G, cols, r, sig, start=()):
    sel = list(start)
    best = cv_rmse(np.column_stack([G] + [cols[c] for c in sel]), r, sig)
    while True:
        trial = sorted((cv_rmse(np.column_stack([G] + [cols[c] for c in sel + [cand]]), r, sig), cand)
                       for cand in GRID if cand not in sel)
        if not trial or trial[0][0] > best - EPS:
            return sel
        best = trial[0][0]
        sel.append(trial[0][1])


def main():
    from train_lib import Data
    d = Data()
    tr, te = load()
    P, H = holdout_split(len(tr))
    kt, _ = group_keys(tr, te, d.subn, d.subnt)
    kP, kH = tuple(k[P] for k in kt), tuple(k[H] for k in kt)
    is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
    mem = sorted(f[:-4] for f in os.listdir("exp/hold/pred") if os.path.getsize(f"exp/hold/pred/{f}") > 0)
    gas = [X8.get(m, m) for m in mem if not is_el(m)] + ["p1_gpm"]
    el = [m for m in mem if is_el(m)] + ["p1_lepm"]
    path = lambda n, part: next(f for f in (f"exp/hold/{part}/{n}.npy", f"exp/hold/tune/{part}/{n}.npy")  # noqa: E731
                                if os.path.exists(f))
    A = tr["bld_dim_06"].values
    M = tr["bld_dim_12"].values.astype(float)
    halves = np.random.default_rng(11).permutation(len(H)) % 2
    final = {}
    for metric, col in METRICS.items():
        if FINAL is not None and metric not in FINAL:
            final[metric] = {"configs": [list(c) for c in CONFIGS[metric]]}
            continue
        names = el if metric == "elec" else gas
        Po = np.column_stack([np.load(path(n, "oof")) for n in names])
        Ph = np.column_stack([np.load(path(n, "pred")) for n in names])
        yv = tr[EL if metric == "elec" else EUI].values
        sc = A[P] if metric == "gas" else None
        pP, pH, _ = blend_vc(Po, Ph, yv[P], SPEC, dict(A=A[P], M=M[P]), dict(A=A[H], M=M[H]), sc)
        if metric == "gas":
            pP, pH = pP * A[P], pH * A[H]
        scale = A if metric == "gas" else np.ones(len(A))
        rP, rH = scale[P] * yv[P] - pP, scale[H] * yv[H] - pH
        G = features(metric, kP, kH, rP, pP, pH, A[P], A[H])
        cols = {c: np.load(f"{CACHE}/{metric}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in GRID}
        sig = SIGMAS[col]
        cur = [tuple(c) for c in CONFIGS[metric]]
        if FINAL is not None:
            sel = greedy(G, cols, rH, sig, cur)
            final[metric] = {"configs": [list(c) for c in sel]}
            print(f"{metric} final: current {cv_rmse(np.column_stack([G] + [cols[c] for c in cur]), rH, sig):.6f} -> "
                  f"{cv_rmse(np.column_stack([G] + [cols[c] for c in sel]), rH, sig):.6f} with +{sel[len(cur):]}",
                  flush=True)
            continue
        out = []
        for h in (0, 1):
            a, b = halves == h, halves != h
            sel = greedy(G[a], {c: v[a] for c, v in cols.items()}, rH[a], sig, cur if AUG else ())
            new = cv_rmse(np.column_stack([G[b]] + [cols[c][b] for c in sel]), rH[b], sig)
            old = cv_rmse(np.column_stack([G[b]] + [cols[c][b] for c in cur]), rH[b], sig)
            out.append((new, old, len(sel)))
            print(f"{metric} select on half {h}, score other half: reselected {new:.5f} ({len(sel)} cfg) "
                  f"vs current {old:.5f} -> {new - old:+.6f}", flush=True)
        print(f"{metric} mean honest change {np.mean([o[0] - o[1] for o in out]):+.6f}", flush=True)
    if FINAL is not None:
        json.dump(final, open("exp/analysis/corr_final.json", "w"), indent=1)
        print("wrote exp/analysis/corr_final.json", flush=True)


if __name__ == "__main__":
    main()
