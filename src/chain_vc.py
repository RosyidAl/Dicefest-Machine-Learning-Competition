"""Holdout chain (src/resid_knn.py fit) with a varying-coefficient blend: member weights piecewise-linear in the ranks
of floor area A (bld_dim_06), meters M (bld_dim_12) and / or the global blend prediction p (src/blend.py blend_vc).
Same hold members, correction features and cross-fitted lambda as resid_knn.fit, so the H scores compare directly
with exp/resid_knn_fit_v19.log.

usage: python src/chain_vc.py <spec> <lam_out.json> [configs.json]
       spec: one for all metrics (e.g. A2xp2) or gas=A2xp2,eui=A2xM2,elec=p2 ; a bare integer n = A<n>
       configs.json: kNN correction configs per metric, as {metric: {"configs": [...]}} (exp/analysis/select_corr.json)
       or {"configs": {metric: [...]}} (a resid_knn json); default src/resid_knn.py CONFIGS
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from blend import blend_vc  # noqa: E402
from common import SIGMAS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402
from resid_corr import EL, EUI, METRICS, X8, group_keys  # noqa: E402
from resid_knn import CONFIGS, KSHR, all_features, spaces  # noqa: E402


def parse_specs(arg):
    if arg.isdigit():
        arg = f"A{arg}"
    if "=" not in arg:
        return {m: arg for m in METRICS}
    return dict(p.split("=") for p in arg.split(","))


def load_configs(path):
    if not path:
        return {m: [tuple(c) for c in CONFIGS[m]] for m in METRICS}
    js = json.load(open(path))
    return {m: [tuple(c) for c in (js["configs"][m] if "configs" in js else js[m]["configs"])] for m in METRICS}


def main(spec_arg, lam_file, cfg_file=None):
    from train_lib import Data
    specs = parse_specs(spec_arg)
    configs = load_configs(cfg_file)
    print("specs", specs, "configs", cfg_file or "CONFIGS", flush=True)
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
    lam = {}
    for metric, col in METRICS.items():
        names = el if metric == "elec" else gas
        Po = np.column_stack([np.load(path(n, "oof")) for n in names])
        Ph = np.column_stack([np.load(path(n, "pred")) for n in names])
        yv = tr[EL if metric == "elec" else EUI].values
        sc = A[P] if metric == "gas" else None
        pP, pH, _ = blend_vc(Po, Ph, yv[P], specs[metric], dict(A=A[P], M=M[P]), dict(A=A[H], M=M[H]), sc)
        if metric == "gas":
            pP, pH = pP * A[P], pH * A[H]
        scale = A if metric == "gas" else np.ones(len(A))
        rP, rH = scale[P] * yv[P] - pP, scale[H] * yv[H] - pH
        F = all_features(metric, kP, kH, rP, pP, pH, A[P], A[H], Zs, P, H, d.subn[P], d.subn[H], configs[metric])
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
    json.dump({"lam": lam, "configs": configs, "kshr": KSHR, "vc": specs}, open(lam_file, "w"), indent=1)
    print("wrote", lam_file)


if __name__ == "__main__":
    main(*sys.argv[1:4])
