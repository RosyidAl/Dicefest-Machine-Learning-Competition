"""Audit postproc 07: are rows with very few meters (M = bld_dim_12 <= 3, 4-5) systematically under-predicted?

- counts of M values in train / test;
- mean relative residual (y - p) / p by M, on H (corrected and teacher) and on P-OOF (teacher, independent rows);
- honest gain of a per-M-group relative shift p' = p * (1 + delta_g):
    (a) delta fitted on P-OOF teacher residuals, applied to H corrected (fit and score on disjoint rows);
    (b) delta 5-fold cross-fitted on H;
  groups: {M<=3} only, and {M<=3},{M in 4..5} separately.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/07_smallM.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
MP, MH = Mall[P], Mall[H]
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_12"])
Mt = te["bld_dim_12"].values
print("M counts train:", {int(k): int((Mall == k).sum()) for k in range(2, 8)})
print("M counts test :", {int(k): int((Mt == k).sum()) for k in range(2, 8)})
print("M counts H    :", {int(k): int((MH == k).sum()) for k in range(2, 8)})


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


tot = {}
for m in ("gas", "eui", "elec"):
    yH, pc, pt = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None), z[f"{m}_teacher"]
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    print(f"== {m}")
    for k in (2, 3, 4, 5, 6, 7, 8):
        sh, sp = MH == k, MP == k
        rel = lambda y, p: (y - p) / p  # noqa: E731
        rc, rt, rp = rel(yH[sh], pc[sh]), rel(yH[sh], pt[sh]), rP[sp] / pP[sp]
        print(f"   M={k}: H n {sh.sum():5d} corrected rel {rc.mean():+.4f} (se {rc.std() / np.sqrt(max(sh.sum(), 1)):.4f}) "
              f"teacher rel {rt.mean():+.4f} | P-OOF n {sp.sum():5d} rel {rp.mean():+.4f} (se {rp.std() / np.sqrt(max(sp.sum(), 1)):.4f})")
    base = nr(yH, pc, m)
    for gname, groups in (("M<=3", [(2, 3)]), ("M<=3 & M4-5", [(2, 3), (4, 5)]), ("M=2,M=3 separate", [(2, 2), (3, 3)])):
        # (a) fit on P-OOF teacher residuals (least squares of r on p within group), apply to H corrected
        pa = pc.copy()
        deltas = []
        for lo, hi in groups:
            g = (MP >= lo) & (MP <= hi)
            d = float(pP[g] @ rP[g] / (pP[g] @ pP[g]))
            deltas.append(round(d, 4))
            gh = (MH >= lo) & (MH <= hi)
            pa[gh] = pc[gh] * (1 + d)
        da = nr(yH, pa, m) - base
        # (b) 5-fold cross-fitted on H
        pb = pc.copy()
        for a, b in KFold(5, shuffle=True, random_state=0).split(pc):
            for lo, hi in groups:
                ga = a[(MH[a] >= lo) & (MH[a] <= hi)]
                gb = b[(MH[b] >= lo) & (MH[b] <= hi)]
                d = float(pc[ga] @ (yH[ga] - pc[ga]) / (pc[ga] @ pc[ga]))
                pb[gb] = pc[gb] * (1 + d)
        db = nr(yH, pb, m) - base
        print(f"   {gname:18s} (a) P-OOF deltas {deltas} -> H d {da:+.7f} total {da / 5:+.8f} | (b) H cross-fit d {db:+.7f} total {db / 5:+.8f}")
        tot[(gname, "a")] = tot.get((gname, "a"), 0) + da / 5
        tot[(gname, "b")] = tot.get((gname, "b"), 0) + db / 5
print("TOTAL", {f"{k[0]} {k[1]}": round(v, 8) for k, v in tot.items()})
