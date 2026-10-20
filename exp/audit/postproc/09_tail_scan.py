"""Audit postproc 09: scan for systematic relative bias in the tails of simple size/count features (as found for
M <= 4 in 07). Relative residual sum(r)/sum(p) per quantile bin of each feature, on two independent samples:
P-OOF teacher residuals (300k rows) and H corrected residuals (129k rows). A bin is flagged when both samples agree
in sign with |t| >= 2.5.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/09_tail_scan.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H = z["P"], z["H"]
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_01", "bld_dim_02", "bld_dim_03", "bld_dim_06",
                                                     "bld_dim_12", "bld_dim_17"])
A, M, E = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float), tr["bld_dim_01"].values.astype(float)
feats = {"M": M, "A": A, "E": E, "A/M": A / M, "E/M": E / M, "A/E": A / E, "b03/A": tr["bld_dim_03"].values / A,
         "b17": tr["bld_dim_17"].values.astype(float), "clean": tr["bld_dim_02"].values / E}
qs = [0, .0005, .001, .005, .01, .05, .5, .95, .99, .995, .999, .9995, 1]


def rel_stats(r, p):
    """relative bias sum(r)/sum(p) and a t-stat from per-row relative residuals"""
    x = r / p
    return r.sum() / p.sum(), x.mean() / (x.std() / np.sqrt(len(x)) + 1e-12)


for m in ("gas", "eui", "elec"):
    rP, pP = z[f"{m}_rP"], z[f"{m}_pP"]
    pH = np.clip(z[f"{m}_corrected"], 0, None)
    rH = z[f"{m}_y"] - pH
    print(f"== {m}")
    for fn, f in feats.items():
        e = np.unique(np.quantile(f, qs))
        for i in range(len(e) - 1):
            last = i == len(e) - 2
            sp = (f[P] >= e[i]) & ((f[P] < e[i + 1]) | (last & (f[P] <= e[i + 1])))
            sh = (f[H] >= e[i]) & ((f[H] < e[i + 1]) | (last & (f[H] <= e[i + 1])))
            if sp.sum() < 5 or sh.sum() < 5:
                continue
            bp, tp = rel_stats(rP[sp], pP[sp])
            bh, th = rel_stats(rH[sh], pH[sh])
            flag = "  <==" if (np.sign(bp) == np.sign(bh) and abs(tp) >= 2.5 and abs(th) >= 2.5) else ""
            if flag or i in (0, len(e) - 2):
                print(f"   {fn:6s} [{e[i]:.4g},{e[i + 1]:.4g}] P n {sp.sum():6d} rel {bp:+.4f} t {tp:+5.1f} | "
                      f"H n {sh.sum():6d} rel {bh:+.4f} t {th:+5.1f}{flag}")
