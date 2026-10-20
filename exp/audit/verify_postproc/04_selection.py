"""Verify postproc S1, step 4: selection / specification sensitivity on the honest H chain.

- groups chosen by a rule that sees only P-OOF (include M group g, for g in 2..8 one at a time, M<=3 pooled, when the
  P-OOF LS delta has |t| > 3 for at least 2 of the 3 metrics), then scored on H;
- alternative group definitions ((2..4) pooled; (2,3),(4),(5); M<=3 only) with P-OOF deltas;
- shrinkage of the delta: q = p (1 + c d), c in 0.5..1.5 (risk / gain trade-off);
- split of the expected test change into the M2-3 and M4 parts (analytic, submission deltas).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/04_selection.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
COL = {"gas": GAS, "eui": EUI, "elec": EL}
SIG = {m: SIGMAS[c] for m, c in COL.items()}
MET = ("gas", "eui", "elec")
zf = np.load("exp/audit/chain_H.npz", allow_pickle=False)
z = {k: zf[k] for k in zf.files if not k.endswith("_F")}
P, H, Mall = z["P"], z["H"], z["M"]
MP, MH = Mall[P], Mall[H]
y = {m: z[f"{m}_y"] for m in MET}
pc = {m: np.clip(z[f"{m}_corrected"], 0, None) for m in MET}


def nr(m, q):
    return float(np.sqrt(np.mean((y[m] - q) ** 2)) / SIG[m])


def pstat(m, lo, hi):
    g = (MP >= lo) & (MP <= hi)
    p, r = z[f"{m}_pP"][g], z[f"{m}_rP"][g]
    d = float(p @ r / (p @ p))
    se = float(np.sqrt(np.sum(p ** 2 * (r - d * p) ** 2)) / (p @ p))
    return d, d / se


def total(groups, c=1.0):
    tot, per = 0.0, {}
    for m in MET:
        q = pc[m].copy()
        for lo, hi in groups:
            d, _ = pstat(m, lo, hi)
            g = (MH >= lo) & (MH <= hi)
            q[g] = pc[m][g] * (1 + c * d)
        per[m] = nr(m, q) - nr(m, pc[m])
        tot += per[m] / 5
    return tot, per


print("P-OOF LS delta (t, HC0) by M group:")
cand = [(2, 3)] + [(k, k) for k in range(4, 9)]
chosen = []
for lo, hi in cand:
    st = {m: pstat(m, lo, hi) for m in MET}
    sel = sum(abs(t) > 3 for _, t in st.values()) >= 2
    print(f"   M {lo}-{hi}: " + " ".join(f"{m} {d:+.4f} (t {t:+.1f})" for m, (d, t) in st.items()) + ("  -> selected" if sel else ""))
    if sel:
        chosen.append((lo, hi))
t, per = total(chosen)
print(f"P-only rule picks {chosen}: H total {t:+.8f}", {m: round(v, 7) for m, v in per.items()})
for name, groups in (("claimed (2,3),(4)", [(2, 3), (4, 4)]), ("M<=3 only", [(2, 3)]), ("(2..4) pooled", [(2, 4)]),
                     ("(2,3),(4),(5)", [(2, 3), (4, 4), (5, 5)])):
    t, per = total(groups)
    print(f"   {name:20s}: H total {t:+.8f}", {m: round(v, 7) for m, v in per.items()})
print("shrinkage c (q = p (1 + c d)), claimed groups:")
for c in (0.5, 0.75, 1.0, 1.25, 1.5):
    t, _ = total([(2, 3), (4, 4)], c)
    print(f"   c {c:4.2f}: H total {t:+.8f}")

# expected test change split by group (analytic, submission deltas from postproc/12)
fo = json.load(open("exp/audit/postproc/12_fullvc_smallM.json"))
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_12"])
Mt = te["bld_dim_12"].values.astype(float)
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=[GAS, EL, EUI], float_precision="round_trip")
nrm = {m: nr(m, pc[m]) for m in MET}
for gname, key, (lo, hi) in (("M2-3", "M<=3", (2, 3)), ("M4", "M=4", (4, 4))):
    g = (Mt >= lo) & (Mt <= hi)
    tot = sum(-(fo[m][key]["ls_delta"] ** 2) * np.sum(sub[COL[m]].values[g] ** 2) / len(Mt) / (2 * SIG[m] ** 2 * nrm[m]) / 5
              for m in MET)
    print(f"expected test change from {gname} rows only: {tot:+.8f}")
