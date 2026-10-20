"""verify_postproc/combo c05: row-level robustness of S1 (the part of the gain carried by few rows).

(a) H (honest chain, corrected): for M 2-3 and M 4 rows, the relative bias by prediction tercile and the share of rows
    whose squared error improves under the P-OOF-delta shift; same tercile bias on P-OOF (independent rows) and on the
    full-train VC OOF.
(b) test vs train small-M rows: are the test rows' predictions (sub_v19vce) in the same range as the train rows'
    full-OOF predictions (gas per meter, EUI, elec per meter, A per meter)?
(c) leave-one-out sensitivity on the full OOF: S1 delta for M 2-3 recomputed leaving out each train row in turn
    (how much does one row move the test-side delta?).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/combo/c05_smallM_rows.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/combo"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
Pi, Hi, A, M = z["P"], z["H"], z["A"], z["M"]
MP, MH = M[Pi], M[Hi]
f = np.load(f"{OUT}/c02_fullvc.npz")
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EL, EUI])
Y = {"gas": tr[GAS].values, "eui": tr[EUI].values, "elec": tr[EL].values}
GROUPS = (("M2-3", 2, 3), ("M4", 4, 4))


def terc(p, r):
    e = np.quantile(p, [1 / 3, 2 / 3])
    out = []
    for lo, hi in ((-np.inf, e[0]), (e[0], e[1]), (e[1], np.inf)):
        g = (p > lo) & (p <= hi)
        out.append(f"{r[g].sum() / p[g].sum():+.3f}(n{int(g.sum())})")
    return " ".join(out)


for m in MET:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    yH, pH = z[f"{m}_y"], np.maximum(z[f"{m}_corrected"], 0)
    pf = np.maximum(f[f"{m}_oof"], 0)
    for name, lo, hi in GROUPS:
        gP, gH, gF = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi), (M >= lo) & (M <= hi)
        d = float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
        e0 = (yH[gH] - pH[gH]) ** 2
        e1 = (yH[gH] - pH[gH] * (1 + d)) ** 2
        print(f"{m:4s} {name}: rel bias by p tercile  P-OOF {terc(pP[gP], rP[gP])} | H {terc(pH[gH], yH[gH] - pH[gH])} | "
              f"full OOF {terc(pf[gF], Y[m][gF] - pf[gF])} || H rows improved {np.mean(e1 < e0):.2f} of {int(gH.sum())}, "
              f"sum dSE/sigma^2 {np.sum(e1 - e0) / SIGMAS[COL[m]] ** 2:+.3f}")

# (b) test vs train small-M prediction ranges
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
for name, lo, hi in GROUPS:
    g, gt = (M >= lo) & (M <= hi), (Mt >= lo) & (Mt <= hi)
    q = lambda v: np.round(np.quantile(v, [0.1, 0.5, 0.9]), 1).tolist()  # noqa: E731
    print(f"(b) {name}: A/M train {q(A[g] / M[g])} test {q(At[gt] / Mt[gt])} | gas/M pred train-OOF "
          f"{q(f['gas_oof'][g] / M[g])} test {q(sub[GAS].values[gt] / Mt[gt])} | EUI pred train-OOF {q(f['eui_oof'][g])} "
          f"test {q(sub[EUI].values[gt])} | elec/M pred train-OOF {q(f['elec_oof'][g] / M[g])} test {q(sub[EL].values[gt] / Mt[gt])}")

# (c) leave-one-out influence on the full-OOF M 2-3 delta
for m in MET:
    pf = np.maximum(f[f"{m}_oof"], 0)
    g = np.where((M >= 2) & (M <= 3))[0]
    p, r = pf[g], Y[m][g] - pf[g]
    num, den = p @ r, p @ p
    loo = (num - p * r) / (den - p * p)
    print(f"(c) {m:4s} M2-3 delta {num / den:.4f}; leave-one-out range [{loo.min():.4f}, {loo.max():.4f}]; "
          f"half-sample (200 random halves) sd "
          f"{np.std([(lambda s: (p[s] @ r[s]) / (p[s] @ p[s]))(np.random.default_rng(i).permutation(len(g))[: len(g) // 2]) for i in range(200)]):.4f}")
