"""verify2_postproc/combo_repro c5: placebo / null distribution for the S1 cross-fit machinery on the full-train OOF.
Replace the two real groups (M 2-3: 136 rows, M 4: 741 rows) by random groups of the same sizes drawn from rows with
M >= 6 and the same rank band of the prediction (so the prediction scale is comparable), then run the identical
5-fold cross-fitted S1 (LS multiplicative delta per group from the fit folds, applied to the held-out fold).
If the machinery is honest, the placebo total change is about 0 or slightly positive; the real S1 total from
c2 (-0.0000436) is compared with this null.
usage (from /root/Dicefest): OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
    exp/audit/verify2_postproc/combo_repro/c5_placebo.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

D = "exp/audit/verify2_postproc/combo_repro"
TC = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
MET = ("gas", "eui", "elec")
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_12"] + TC)
M = tr["bld_dim_12"].values.astype(float)
y = {"gas": tr[TC[0]].values, "elec": tr[TC[1]].values, "eui": tr[TC[2]].values}
t = np.load(f"{D}/c2_teacher_oof.npz")
p0 = {m: t[m] for m in MET}
n = len(M)
rm0 = {m: np.sqrt(np.mean((y[m] - p0[m]) ** 2)) for m in MET}


def s1_total(groups, seed=0):
    tot = 0.0
    for m in MET:
        p = p0[m].copy()
        for fit, app in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
            fitmask = np.zeros(n, bool)
            fitmask[fit] = True
            for g in groups:
                gf, ga = g[fitmask[g]], g[~fitmask[g]]
                d = float(p0[m][gf] @ (y[m][gf] - p0[m][gf]) / (p0[m][gf] @ p0[m][gf]))
                p[ga] = p0[m][ga] * (1 + d)
        tot += (np.sqrt(np.mean((y[m] - p) ** 2)) - rm0[m]) / SIG[m] / 5
    return tot


real = s1_total([np.where((M >= 2) & (M <= 3))[0], np.where(M == 4)[0]])
print(f"real S1 (5-fold seed0) total {real:+.8f}")
rng = np.random.default_rng(2026)
pool = np.where(M >= 6)[0]
# match the prediction scale: draw placebo rows within the 5-95% band of the small-M rows' EUI predictions
small = np.where(M <= 4)[0]
lo, hi = np.quantile(p0["eui"][small], [0.05, 0.95])
pool = pool[(p0["eui"][pool] >= lo) & (p0["eui"][pool] <= hi)]
null = []
for it in range(200):
    g1 = rng.choice(pool, 136, replace=False)
    rest = np.setdiff1d(pool, g1)
    g2 = rng.choice(rest, 741, replace=False)
    null.append(s1_total([np.sort(g1), np.sort(g2)], seed=it))
null = np.array(null)
pval = float((np.sum(null <= real) + 1) / (len(null) + 1))
print(f"placebo (200 draws): mean {null.mean():+.8f} sd {null.std():.8f} min {null.min():+.8f} q05 {np.quantile(null, 0.05):+.8f} "
      f"| real {real:+.8f}, empirical p {pval:.4f}")
json.dump(dict(real=real, null_mean=float(null.mean()), null_sd=float(null.std()), null_min=float(null.min()),
               p=pval), open(f"{D}/c5_placebo.json", "w"), indent=1)
