"""Independent confirmation on the deployed members: global NNLS blend of the v19 members (weights from
final/work/v19vce/summary.json) on the full-train OOF arrays (exp/oof). Residual bias for small M and near the gas cap,
plus honest 2-fold estimate of the two fixes on full OOF (parameters fitted on one half of the rows, scored on the
other; teacher level, i.e. without the residual correction)."""
import json

import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
s = json.load(open("final/work/v19vce/summary.json"))
T = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
tr = pd.read_parquet("cache/train.parquet", columns=list(T.values()) + ["bld_dim_06", "bld_dim_12"])
A, M = tr.bld_dim_06.values, tr.bld_dim_12.values.astype(float)
pred = {}
for m in ("gas", "eui", "elec"):
    w = s["weights"][m]
    p = sum(np.load(f"exp/oof/{k}.npy") * v for k, v in w.items() if v > 0)
    pred[m] = p * A if m == "gas" else p
    y = tr[T[m]].values
    print(f"{m}: global-blend OOF NRMSE {np.sqrt(np.mean((y - pred[m]) ** 2)) / SIG[m]:.6f}", flush=True)
for m in pred:
    y = tr[T[m]].values
    r = y - pred[m]
    for v in (3, 4, 5, 6):
        sl = M == v
        print(f"  {m:4s} M={v}: n={sl.sum():6d} rel bias {r[sl].sum() / pred[m][sl].sum():+.4f} t {r[sl].mean() / (r[sl].std() / np.sqrt(sl.sum())):+.1f}")
g, yg = pred["gas"], tr[T["gas"]].values
for lo, hi in ((8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e7)):
    sl = (g >= lo) & (g < hi)
    print(f"  gas pred in [{lo:.3g},{hi:.3g}): n={sl.sum()} mean resid {np.mean(yg[sl] - g[sl]):+.0f}")
# honest 2-fold on full OOF (teacher level)
rng = np.random.default_rng(3)
half = rng.permutation(len(A)) % 2
tot = 0.0
for m in pred:
    y, p = tr[T[m]].values, pred[m]
    q = p.copy()
    for h in (0, 1):
        a, b = half == h, half != h
        if m in ("gas", "eui"):
            ga, gb = (p[a], p[b]) if m == "gas" else (p[a] * A[a], p[b] * A[b])
            yg_a = yg[a]
            hh = np.maximum(ga - 9e5, 0)
            beta = hh @ (yg_a - ga) / (hh @ hh)
            fixed = np.minimum(gb + beta * np.maximum(gb - 9e5, 0), 1e6)
            q[b] = fixed if m == "gas" else np.where(gb > 9e5, fixed / A[b], p[b])
        for grp in ((M <= 3), (M == 4)):
            fa, fb = grp & a, grp & b
            f = (y[fa] @ p[fa]) / (p[fa] @ p[fa])
            q[fb] = q[fb] * f
    dd = (np.sqrt(np.mean((y - q) ** 2)) - np.sqrt(np.mean((y - p) ** 2))) / SIG[m]
    tot += dd / 5
    print(f"{m}: 2-fold honest change of G+S on full OOF (teacher level) {dd:+.7f} total {dd / 5:+.8f}")
print(f"TOTAL {tot:+.8f}")
