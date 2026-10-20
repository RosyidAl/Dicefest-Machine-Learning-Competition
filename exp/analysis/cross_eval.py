"""ENS0020: gas <-> EUI consistency inside the residual correction. The gas blend (loss weighted by A^2) and the EUI blend
(unweighted) are two estimates of the same quantity (gas = A * EUI). Holdout chain as pl_eval.py: add the other
blend's estimate minus this blend's estimate as one more correction regressor, cross-fitted 5-fold on H.

usage: python exp/analysis/cross_eval.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "exp/analysis")
from corr_est import chain  # noqa: E402
from resid_corr import METRICS  # noqa: E402
from select_corr import CACHE, cv_rmse  # noqa: E402


def main():
    tr, te, d, P, H, A, res = chain()
    cfgs = {m: [tuple(c) for c in v] for m, v in json.load(open("exp/resid_knn_vc_AMp_e.json"))["configs"].items()}
    AH = A[H]
    other = {"gas": res["eui"]["pH"] * AH - res["gas"]["pH"], "eui": res["gas"]["pH"] / AH - res["eui"]["pH"]}
    for metric in ("gas", "eui"):
        R = res[metric]
        F = np.column_stack([R["G"]] + [np.load(f"{CACHE}/{metric}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in cfgs[metric]])
        base = cv_rmse(F, R["rH"], R["sig"])
        x = other[metric]
        v = cv_rmse(np.column_stack([F, x]), R["rH"], R["sig"])
        # the same with a size interaction (the better estimate depends on floor area)
        r = np.argsort(np.argsort(AH)) / len(AH)
        v2 = cv_rmse(np.column_stack([F, x, x * r]), R["rH"], R["sig"])
        coef = np.linalg.lstsq(np.column_stack([F, x, x * r]), R["rH"], rcond=None)[0][-2:]
        print(f"{metric}: corrected {base:.6f}  +cross {v:.6f} ({v - base:+.6f})  +cross x rank(A) {v2:.6f} ({v2 - base:+.6f}) "
              f"coef {np.round(coef, 3)}", flush=True)


if __name__ == "__main__":
    main()
