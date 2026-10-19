"""Holdout chain (src/resid_knn.py fit) with the size-varying blend: member weights piecewise-linear in the rank of
floor area (bld_dim_06) over the fit rows (src/blend.py size_basis / nnls_blend_vc). Same hold members, correction
features and cross-fitted lambda as resid_knn.fit, so the H scores compare directly with exp/resid_knn_fit_v19.log.

usage: python src/chain_vc.py <n_knots> <lam_out.json>
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from blend import nnls_blend_vc, predict_vc, size_basis  # noqa: E402
from common import SIGMAS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402
from resid_corr import EL, EUI, METRICS, X8, group_keys  # noqa: E402
from resid_knn import CONFIGS, KSHR, all_features, spaces  # noqa: E402


def blend_vc(Pf, Pa, yf, metric, A_fit, A_app, knots):
    Bf, Ba = size_basis(A_fit, A_fit, knots), size_basis(A_fit, A_app, knots)
    W = nnls_blend_vc(Pf, yf, Bf, A_fit if metric == "gas" else None)
    s_f, s_a = (A_fit, A_app) if metric == "gas" else (1.0, 1.0)
    return predict_vc(Pf, W, Bf) * s_f, predict_vc(Pa, W, Ba) * s_a


def main(n_knots, lam_file):
    from train_lib import Data
    knots = np.linspace(0, 1, n_knots)
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
    lam = {}
    for metric, col in METRICS.items():
        names = el if metric == "elec" else gas
        Po = np.column_stack([np.load(path(n, "oof")) for n in names])
        Ph = np.column_stack([np.load(path(n, "pred")) for n in names])
        yv = tr[EL if metric == "elec" else EUI].values
        pP, pH = blend_vc(Po, Ph, yv[P], metric, A[P], A[H], knots)
        scale = A if metric == "gas" else np.ones(len(A))
        rP, rH = scale[P] * yv[P] - pP, scale[H] * yv[H] - pH
        F = all_features(metric, kP, kH, rP, pP, pH, A[P], A[H], Zs, P, H, d.subn[P], d.subn[H])
        base = np.sqrt(np.mean(rH ** 2)) / SIGMAS[col]
        msg = []
        for nf in (3, F.shape[1]):
            pc = np.zeros(len(H))
            for a, b in KFold(5, shuffle=True, random_state=0).split(F):
                pc[b] = F[b, :nf] @ np.linalg.lstsq(F[a, :nf], rH[a], rcond=None)[0]
            msg.append(f"{np.sqrt(np.mean((rH - pc) ** 2)) / SIGMAS[col]:.5f}")
        lam[metric] = np.linalg.lstsq(F, rH, rcond=None)[0].tolist()
        print(f"{metric:4s} H {base:.5f} -> group {msg[0]} -> group+knn {msg[1]}  lam {np.round(lam[metric], 3)}",
              flush=True)
    json.dump({"lam": lam, "configs": CONFIGS, "kshr": KSHR, "vc_knots": knots.tolist()}, open(lam_file, "w"), indent=1)
    print("wrote", lam_file)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
