"""verify2_integrity: a different honest split. Full-train 10-fold OOF predictions of the 27 v19vce gas-family members
(models trained on 90% of all train rows, independent of the chain_H models), blended with the production
varying-coefficient NNLS (final/src/blend.blend_vc, spec A2xM2xp2) and with a global NNLS, both CROSS-FITTED on two
random halves of train (seed 2028): weights fitted on one half, applied to the other. Then the fit-free cap
gas <= 1e6, EUI <= 1e6/A is scored on all 429,303 rows. Bound also taken honestly from the fit half only."""
import json
import sys
import time
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "final/src")
from blend import blend_vc, nnls_blend  # noqa: E402  (library code of the production pipeline)

t0 = time.time()
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
members = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12"])
g, ye, A, M = tr[GAS].values, tr[EUI].values, tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
Po = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in members])
print("loaded", Po.shape, round(time.time() - t0, 1), flush=True)
n = len(g)
half = np.random.default_rng(2028).random(n) < 0.5
folds = [(np.where(half)[0], np.where(~half)[0]), (np.where(~half)[0], np.where(half)[0])]
pred = {k: np.zeros(n) for k in ["vc_gas", "vc_eui", "gl_gas", "gl_eui"]}
capfit = np.zeros(n)  # bound from the fit half's labels only
for f, a in folds:
    capfit[a] = g[f].max()
    _, pa, _ = blend_vc(Po[f], Po[a], ye[f], "A2xM2xp2", dict(A=A[f], M=M[f]), dict(A=A[a], M=M[a]), A[f])
    pred["vc_gas"][a] = pa * A[a]
    _, pa, _ = blend_vc(Po[f], Po[a], ye[f], "A2xM2xp2", dict(A=A[f], M=M[f]), dict(A=A[a], M=M[a]), None)
    pred["vc_eui"][a] = pa
    pred["gl_gas"][a] = Po[a] @ nnls_blend(Po[f], ye[f], A[f]) * A[a]
    pred["gl_eui"][a] = Po[a] @ nnls_blend(Po[f], ye[f])
    print("fold done", round(time.time() - t0, 1), flush=True)
for k in pred:
    pred[k] = np.clip(pred[k], 0, None)  # the production >= 0 clip


def nr(y, p, s):
    return float(np.sqrt(np.mean((y - p) ** 2)) / s)


out = {"members": len(members), "fit_half_bounds": sorted(set(capfit.tolist()))}
for blend in ["vc", "gl"]:
    for cname, cg in [("1e6", np.full(n, 1e6)), ("fit_half_max", capfit)]:
        tot = 0.0
        for m, y, cap in [("gas", g, cg), ("eui", ye, cg / A)]:
            p = pred[f"{blend}_{m}"]
            q = np.minimum(p, cap)
            k = p > cap
            d = (nr(y, q, SIG[m]) - nr(y, p, SIG[m])) / 5
            dsq = (y - q) ** 2 - (y - p) ** 2
            # split-half view: the two cross-fit halves separately
            dh = [(nr(y[i], q[i], SIG[m]) - nr(y[i], p[i], SIG[m])) / 5 for i in (np.where(half)[0], np.where(~half)[0])]
            out[f"{blend}_{cname}_{m}"] = {"base_nrmse": nr(y, p, SIG[m]), "n_above": int(k.sum()), "delta_total": d,
                                           "halves_delta_total": dh, "n_rows_worse": int((dsq > 1e-9).sum()),
                                           "max_pred": float(p.max()),
                                           "mean_y_minus_p_above": float(np.mean(y[k] - p[k])) if k.any() else None}
            tot += d
        out[f"{blend}_{cname}_total"] = tot
out["seconds"] = round(time.time() - t0, 1)
np.savez_compressed("exp/audit/verify2_integrity/clip_v3_fulloof_preds.npz", **pred, half=half)
json.dump(out, open("exp/audit/verify2_integrity/clip_v3_fulloof.json", "w"), indent=1)
for k, v in out.items():
    print(k, v)
