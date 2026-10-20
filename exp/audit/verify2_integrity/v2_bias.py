"""verify2_integrity / step 2: is the honest chain prediction biased high near the 1e6 gas bound?
Residual means (y - pred) by prediction band, on H (teacher and corrected) and on P (teacher, P-OOF members),
gas on the gas scale and EUI mapped to the gas scale (x A). Independent code, inputs from exp/audit/chain_H.npz.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_integrity/v2_bias.py
"""
import json

import numpy as np

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
bands = [(0, 4e5), (4e5, 6e5), (6e5, 7e5), (7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 2e6)]


def table(p, y):
    out = {}
    for lo, hi in bands:
        m = (p >= lo) & (p < hi)
        r = y[m] - p[m]
        n = int(m.sum())
        out[f"{int(lo / 1e3)}k-{int(hi / 1e3)}k"] = dict(
            n=n, mean_res=round(float(r.mean()), 1) if n else None,
            se=round(float(r.std(ddof=1) / np.sqrt(n)), 1) if n > 1 else None,
            frac_y_ge_p=round(float((r >= 0).mean()), 3) if n else None)
    return out


res = {}
for metric in ("gas", "eui"):
    sH = A[H] if metric == "eui" else 1.0
    sP = A[P] if metric == "eui" else 1.0
    yH = z[f"{metric}_y"] * sH
    yP = z[f"{metric}_pP"] * sP + z[f"{metric}_rP"] * sP
    res[metric] = dict(H_teacher=table(z[f"{metric}_teacher"] * sH, yH),
                       H_corrected=table(z[f"{metric}_corrected"] * sH, yH),
                       P_teacher=table(z[f"{metric}_pP"] * sP, yP))
    res[metric]["P_y_max"] = float(yP.max())
    res[metric]["H_y_max"] = float(yH.max())
json.dump(res, open("exp/audit/verify2_integrity/v2_bias.json", "w"), indent=1)
for metric in res:
    for part in ("H_teacher", "H_corrected", "P_teacher"):
        print(metric, part)
        for k, v in res[metric][part].items():
            print(f"   {k:>12} n={v['n']:>7} mean_res={v['mean_res']} se={v['se']} P(y>=p)={v['frac_y_ge_p']}")
