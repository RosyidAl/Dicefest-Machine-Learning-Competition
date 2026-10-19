"""Re-select the similarity-kNN residual-correction configs for the varying-coefficient blend (D012).

Holdout chain as src/chain_vc.py (all hold members, blend A2xM2xp2 fitted on P-OOF, applied to H). For each metric:
group features (zone / cell / window) + greedy forward selection over a grid of kNN configs (space, W, k, spatial),
scored by the 5-fold cross-fitted least-squares correction RMSE on H. Candidate columns are cached in exp/corrsel/.
The current CONFIGS (src/resid_knn.py) are scored the same way for reference.

usage: python exp/analysis/select_corr.py <out.json>
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from blend import blend_vc  # noqa: E402
from common import SIGMAS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402
from resid_corr import EL, EUI, METRICS, X8, features, group_keys  # noqa: E402
from resid_knn import CONFIGS, knn_mean, spaces  # noqa: E402

SPEC = "A2xM2xp2"
CACHE = "exp/corrsel"
GRID = sorted({(sp, W, k, s) for sp in ("full", "era", "small", "fc", "cen")
               for W, k in ((10, 10), (30, 10), (30, 20), (100, 20), (300, 50), (1000, 100), (3000, 200))
               for s in ((0.0, 4.0) if sp == "full" and W in (30, 100, 300) else (0.0,))})
EPS = 2e-6


def cv_rmse(F, r, sig):
    pc = np.zeros(len(r))
    for a, b in KFold(5, shuffle=True, random_state=0).split(F):
        pc[b] = F[b] @ np.linalg.lstsq(F[a], r[a], rcond=None)[0]
    return np.sqrt(np.mean((r - pc) ** 2)) / sig


def main(out):
    from train_lib import Data
    os.makedirs(CACHE, exist_ok=True)
    d = Data()
    tr, te = load()
    P, H = holdout_split(len(tr))
    kt, _ = group_keys(tr, te, d.subn, d.subnt)
    kP, kH = tuple(k[P] for k in kt), tuple(k[H] for k in kt)
    Zs = spaces(tr, te)
    is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
    mem = sorted(f[:-4] for f in os.listdir("exp/hold/pred") if os.path.getsize(f"exp/hold/pred/{f}") > 0)
    gas = [X8.get(m, m) for m in mem if not is_el(m)] + ["p1_gpm"]
    el = [m for m in mem if is_el(m)] + ["p1_lepm"]
    path = lambda n, part: next(f for f in (f"exp/hold/{part}/{n}.npy", f"exp/hold/tune/{part}/{n}.npy")  # noqa: E731
                                if os.path.exists(f))
    A = tr["bld_dim_06"].values
    M = tr["bld_dim_12"].values.astype(float)
    res = {}
    for metric, col in METRICS.items():
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
        if metric == "gas":
            num, den, mult = rP, A[P], A[H]
        elif metric == "eui":
            num, den, mult = A[P] * rP, A[P], 1.0
        else:
            num, den, mult = rP, pP, pH
        G = features(metric, kP, kH, rP, pP, pH, A[P], A[H])
        cols = {}
        for cfg in GRID:
            f = f"{CACHE}/{metric}_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}.npy"
            if not os.path.exists(f):
                sp, W, k, s = cfg
                np.save(f, knn_mean(Zs[sp][P], d.subn[P], Zs[sp][H], d.subn[H], num, den, W, k, s) * mult)
            cols[cfg] = np.load(f)
        sig = SIGMAS[col]
        cur_cfg = [tuple(c) for c in CONFIGS[metric]]
        ref = cv_rmse(np.column_stack([G] + [cols[c] for c in cur_cfg]), rH, sig)
        sel, best = [], cv_rmse(G, rH, sig)
        print(f"{metric}: H blend {np.sqrt(np.mean(rH ** 2)) / sig:.5f}, group {best:.5f}, current configs {ref:.5f}",
              flush=True)
        while True:
            trial = sorted((cv_rmse(np.column_stack([G] + [cols[c] for c in sel + [cand]]), rH, sig), cand)
                           for cand in GRID if cand not in sel)
            if not trial or trial[0][0] > best - EPS:
                break
            best = trial[0][0]
            sel.append(trial[0][1])
            print(f"   +{trial[0][1]} {best:.5f}", flush=True)
        res[metric] = {"configs": [list(c) for c in sel], "h_rmse": best, "h_rmse_current": ref}
        json.dump(res, open(out, "w"), indent=1)
    print("done", {m: (len(v["configs"]), round(v["h_rmse"], 5), round(v["h_rmse_current"], 5)) for m, v in res.items()})


if __name__ == "__main__":
    main(sys.argv[1])
