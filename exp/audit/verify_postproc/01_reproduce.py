"""Verify postproc S1 (small-M multiplicative shift), step 1: independent re-derivation on the holdout chain.

- alignment of chain_H arrays with the raw train columns (M, A, targets on H; P/H = common.holdout_split);
- deltas = least squares sum(p r) / sum(p^2) on P-OOF teacher residuals for M in {2,3} and M == 4;
- honest change on H corrected (clip >= 0), per metric and total (= sum / 5);
- per-row decomposition of the change (how concentrated is the gain?), leave-top-k-out;
- bootstrap: (i) scoring noise only (Poisson weights on H rows), (ii) + delta estimation noise (Poisson weights on the
  P group rows);
- 5-fold cross-fit on H only (deltas from 4/5 of H corrected residuals);
- the transformation applied to H TEACHER instead of corrected (does the correction interact?).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/01_reproduce.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS, holdout_split  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
MET = ("gas", "eui", "elec")
GROUPS = ((2, 3), (4, 4))

zf = np.load("exp/audit/chain_H.npz", allow_pickle=False)
z = {k: zf[k] for k in zf.files if not k.endswith("_F")}
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]

# ---- alignment checks against the raw train columns
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EL, EUI])
P2, H2 = holdout_split(len(tr))
assert np.array_equal(P, P2) and np.array_equal(H, H2), "P/H differ from common.holdout_split"
assert np.array_equal(A, tr["bld_dim_06"].values) and np.array_equal(Mall, tr["bld_dim_12"].values.astype(float))
for m, col in (("eui", EUI), ("elec", EL)):
    assert np.allclose(z[f"{m}_y"], tr[col].values[H], rtol=1e-12, atol=0), m
    assert np.allclose(z[f"{m}_pP"] + z[f"{m}_rP"], tr[col].values[P], rtol=1e-12, atol=0), m
assert np.allclose(z["gas_y"], tr[GAS].values[H], rtol=1e-9), "gas_y vs gas column"
assert np.allclose(z["gas_pP"] + z["gas_rP"], tr[GAS].values[P], rtol=1e-9)
print("alignment OK: P/H = holdout_split, M/A identical, y on H and P match the raw target columns")
print("max |gas_y - GAS[H]| / GAS:", float(np.max(np.abs(z["gas_y"] - tr[GAS].values[H]) / tr[GAS].values[H])))

MP, MH = Mall[P], Mall[H]
print("counts  P:", {k: int((MP == k).sum()) for k in (1, 2, 3, 4, 5)}, " H:", {k: int((MH == k).sum()) for k in (1, 2, 3, 4, 5)})
y = {m: z[f"{m}_y"] for m in MET}
pc = {m: np.clip(z[f"{m}_corrected"], 0, None) for m in MET}
pt = {m: z[f"{m}_teacher"] for m in MET}
pP = {m: z[f"{m}_pP"] for m in MET}
rP = {m: z[f"{m}_rP"] for m in MET}
gP = [(MP >= lo) & (MP <= hi) for lo, hi in GROUPS]
gH = [(MH >= lo) & (MH <= hi) for lo, hi in GROUPS]


def deltas(m, wP=None):
    out = []
    for g in gP:
        w = np.ones(g.sum()) if wP is None else wP[g]
        p, r = pP[m][g], rP[m][g]
        out.append(float((w * p) @ r / ((w * p) @ p)))
    return out


def apply(m, base, ds):
    q = base.copy()
    for g, d in zip(gH, ds):
        q[g] = base[g] * (1 + d)
    return q


def nr(m, q, w=None):
    e2 = (y[m] - q) ** 2
    return float(np.sqrt(np.mean(e2) if w is None else (w @ e2) / w.sum()) / SIG[m])


D = {m: deltas(m) for m in MET}
print("P-OOF LS deltas (M2-3, M4):", {m: [round(v, 5) for v in D[m]] for m in MET})
tot_c, tot_t = 0.0, 0.0
for m in MET:
    dc = nr(m, apply(m, pc[m], D[m])) - nr(m, pc[m])
    dt = nr(m, apply(m, pt[m], D[m])) - nr(m, pt[m])
    tot_c += dc / 5
    tot_t += dt / 5
    # only M2-3, only M4
    d23 = nr(m, apply(m, pc[m], [D[m][0], 0.0])) - nr(m, pc[m])
    d4 = nr(m, apply(m, pc[m], [0.0, D[m][1]])) - nr(m, pc[m])
    print(f"{m:4s} base corrected {nr(m, pc[m]):.6f}: change {dc:+.7f} (M2-3 only {d23:+.7f}, M4 only {d4:+.7f}) "
          f"| on teacher {dt:+.7f}")
print(f"TOTAL change (P-OOF deltas -> H corrected) {tot_c:+.8f} | on H teacher {tot_t:+.8f}")

# ---- per-row decomposition (first-order contribution to the total score)
print("\nper-row decomposition of the change (total-score units, first order):")
for m in MET:
    q = apply(m, pc[m], D[m])
    N = len(H)
    rmse = np.sqrt(np.mean((y[m] - pc[m]) ** 2))
    c = ((y[m] - q) ** 2 - (y[m] - pc[m]) ** 2) / (N * 2 * rmse * SIG[m]) / 5
    idx = np.where(c != 0)[0]
    cs = np.sort(c[idx])
    print(f"  {m:4s}: rows changed {len(idx)}, sum {c.sum():+.8f}, helped {int((c < 0).sum())}, hurt {int((c > 0).sum())}; "
          f"best 5 {np.round(cs[:5], 8).tolist()}, worst 5 {np.round(cs[-5:], 8).tolist()}")
    for k in (1, 3, 5, 10):
        print(f"       without the {k:2d} most helped rows: {cs[k:].sum():+.8f}")
    if m == "eui":
        top = idx[np.argsort(c[idx])[:8]]
        print("       top EUI rows: M, A, y, p_corr, rel resid, contribution")
        for i in top:
            print(f"         M {MH[i]:.0f} A {A[H][i]:9.1f} y {y[m][i]:9.2f} p {pc[m][i]:9.2f} rel {(y[m][i] - pc[m][i]) / pc[m][i]:+.3f} c {c[i]:+.8f}")

# ---- bootstrap
rng = np.random.default_rng(12345)
B = 2000
tot_s, tot_full = np.zeros(B), np.zeros(B)
e0 = {m: (y[m] - pc[m]) ** 2 for m in MET}
for b in range(B):
    wH = rng.poisson(1.0, len(H)).astype(float)
    wP = rng.poisson(1.0, len(P)).astype(float)
    for m in MET:
        base = np.sqrt((wH @ e0[m]) / wH.sum()) / SIG[m]
        q = apply(m, pc[m], D[m])
        tot_s[b] += (np.sqrt((wH @ ((y[m] - q) ** 2)) / wH.sum()) / SIG[m] - base) / 5
        q2 = apply(m, pc[m], deltas(m, wP))
        tot_full[b] += (np.sqrt((wH @ ((y[m] - q2) ** 2)) / wH.sum()) / SIG[m] - base) / 5
print(f"\nbootstrap (B={B}) total change: scoring-only mean {tot_s.mean():+.8f} sd {tot_s.std():.8f} P(<0) {np.mean(tot_s < 0):.3f}"
      f" | scoring+delta mean {tot_full.mean():+.8f} sd {tot_full.std():.8f} P(<0) {np.mean(tot_full < 0):.3f}")
print("   quantiles scoring+delta 5/50/95%:", np.round(np.quantile(tot_full, [0.05, 0.5, 0.95]), 8).tolist())

# ---- 5-fold cross-fit using H only (deltas from H corrected residuals of 4/5)
tot_cf = 0.0
for m in MET:
    q = pc[m].copy()
    for a, b in KFold(5, shuffle=True, random_state=0).split(q):
        ina, inb = np.zeros(len(H), bool), np.zeros(len(H), bool)
        ina[a], inb[b] = True, True
        for g in gH:
            fa, fb = g & ina, g & inb
            d = float(pc[m][fa] @ (y[m][fa] - pc[m][fa]) / (pc[m][fa] @ pc[m][fa]))
            q[fb] = pc[m][fb] * (1 + d)
    ch = nr(m, q) - nr(m, pc[m])
    tot_cf += ch / 5
    print(f"H-only 5-fold cross-fit {m:4s} change {ch:+.7f}")
print(f"H-only 5-fold cross-fit TOTAL {tot_cf:+.8f}")
