"""Audit postproc 10: is the small-M under-prediction also present in the full-train OOF of the v19vce members (the
models whose test predictions are submitted)? Per-member and global-NNLS-blend relative bias sum(r)/sum(p) for
M <= 3 and M = 4 rows, plus the same on the test side: mean relative difference between the submitted prediction and
a no-shift reference is not available, so we report how many test rows would be touched.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/10_fulloof_smallM.py
"""
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

sys.path.insert(0, "src")
from common import TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EL, EUI])
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
groups = {"M<=3": M <= 3, "M=4": M == 4, "M=5": M == 5, "M>=6": M >= 6}
for fam, names in (("gas", mem["gas"]), ("elec", mem["elec"])):
    X = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
    if fam == "gas":
        targets = {"eui": (tr[EUI].values, np.ones(len(A))), "gas": (tr[EUI].values, A)}
    else:
        targets = {"elec": (tr[EL].values, np.ones(len(A)))}
    for met, (y, sc) in targets.items():
        print(f"== {met} (family {fam}, {len(names)} members)")
        rows = []
        for j, n in enumerate(names):
            p = X[:, j] * sc
            yy = y * sc
            rows.append([n] + [float((yy[g] - p[g]).sum() / p[g].sum()) for g in groups.values()])
        df = pd.DataFrame(rows, columns=["member"] + list(groups))
        print("   member relative bias: median", df[list(groups)].median().round(4).to_dict(),
              "min", df[list(groups)].min().round(4).to_dict(), "max", df[list(groups)].max().round(4).to_dict())
        # global NNLS blend (weights on all rows; for gas rows scaled by A as in nnls_blend row_scale)
        w = nnls(X * sc[:, None], y * sc)[0]
        p = X @ w * sc
        yy = y * sc
        out = {}
        for gname, g in groups.items():
            rel = (yy[g] - p[g]) / p[g]
            out[gname] = (int(g.sum()), round(float((yy[g] - p[g]).sum() / p[g].sum()), 4),
                          round(float(rel.mean() / (rel.std() / np.sqrt(g.sum()))), 1))
        print("   global NNLS blend: group -> (n, rel bias, t)", out)
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_12"])
print("test rows M<=3", int((te["bld_dim_12"].values <= 3).sum()), "M=4", int((te["bld_dim_12"].values == 4).sum()))
