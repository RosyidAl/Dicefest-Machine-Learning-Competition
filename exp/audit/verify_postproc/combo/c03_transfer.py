"""verify_postproc/combo c03: does the S1 + S3 + S4 gain transfer from the holdout regime (models trained on P = 70%)
to the regime of the submitted test predictions (models trained on all train rows)?

Uses the full-train v19vce VC-blend OOF recomputed in c02 (teacher, before the residual correction; c02 showed the
correction moves the small-M groups by <= 0.4% on test and on H, so teacher-level checks are representative).

(a) regime transfer of FIXED holdout-fitted parameters:
    S1 with the P-OOF deltas (chain_H) applied to the full-OOF predictions of the H rows (their labels never touched
    the deltas); S3 with the candidate's k (fitted on H) applied to the full-OOF predictions of the P rows (their
    labels never touched k).
(b) k of S3 refitted on the full OOF (cross-fitted) vs the candidate's k.
(c) fully cross-fitted chain on the full OOF: S1 deltas, S3 (a0, b0, k) and S4 bounds fitted on 9/10 of the train
    rows, applied to the remaining 1/10 (5 repetitions with different fold seeds). Total change on all 429,303 rows.
(d) small-M rows: seen / unseen subzone mix on test vs train, and the OOF bias by seen / unseen.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/combo/c03_transfer.py
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/combo"
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", "subzone_id", GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_12", "subzone_id"])
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
n = len(tr)
Y = {"gas": tr[EUI].values * A, "eui": tr[EUI].values, "elec": tr[EL].values}
assert np.allclose(Y["gas"], tr[GAS].values, rtol=1e-9)
f = np.load(f"{OUT}/c02_fullvc.npz")
PF = {m: np.maximum(f[f"{m}_oof"], 0.0) for m in MET}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
Pi, Hi = z["P"], z["H"]
GROUPS = ((2, 3), (4, 4))
K_CAND = {"gas": 0.5046, "eui": 0.6574}  # 13_candidate.log (k of S3 fitted on all of H after S1)


def nr(m, p, idx):
    return float(np.sqrt(np.mean((Y[m][idx] - p[idx]) ** 2)) / SIG[m])


def deltas(m, idx):
    out = []
    for lo, hi in GROUPS:
        g = idx[(M[idx] >= lo) & (M[idx] <= hi)]
        p, r = PF[m][g], Y[m][g] - PF[m][g]
        out.append(float(p @ r / (p @ p)))
    return out


def apply_s1(m, p, idx, d):
    q = p.copy()
    for (lo, hi), dd in zip(GROUPS, d):
        g = idx[(M[idx] >= lo) & (M[idx] <= hi)]
        q[g] = q[g] * (1 + dd)
    return q


def var_model(m, idx):
    p, r = PF[m][idx], Y[m][idx] - PF[m][idx]
    ok = p > 0
    X = np.column_stack([np.ones(ok.sum()), 1 / M[idx][ok]])
    return np.linalg.lstsq(X, (r[ok] / p[ok]) ** 2, rcond=None)[0]


def s3_x(m, p, idx, vm, gmax):
    a0, b0 = vm
    s = np.maximum(p[idx], 1e-9) * np.sqrt(np.maximum(a0 + b0 / M[idx], 1e-8))
    hi = gmax if m == "gas" else gmax / A[idx]
    b = (hi - p[idx]) / s
    return -s * np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))


summary = {}
# (a) regime transfer with fixed holdout-fitted parameters
dP = {m: [] for m in MET}
for m in MET:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    for lo, hi in GROUPS:
        g = (M[Pi] >= lo) & (M[Pi] <= hi)
        dP[m].append(float(pP[g] @ rP[g] / (pP[g] @ pP[g])))
print("(a) P-OOF deltas", {m: np.round(v, 4).tolist() for m, v in dP.items()},
      "| full-OOF deltas on H rows only (for comparison)", {m: np.round(deltas(m, Hi), 4).tolist() for m in MET})
tot = 0.0
for m in MET:
    q = apply_s1(m, PF[m], Hi, dP[m])
    d = nr(m, q, Hi) - nr(m, PF[m], Hi)
    tot += d / 5
    print(f"   S1 P-OOF deltas -> full-OOF H rows {m:4s}: d {d:+.7f}")
print(f"   S1 regime transfer TOTAL on H rows (full-OOF teacher) {tot:+.8f}   [holdout chain S1: -0.00004577]")
summary["a_S1_fullOOF_Hrows"] = tot
vmP = {}
for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    vmP[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / M[Pi]]), (rP / pP) ** 2, rcond=None)[0]
GMAX_ALL = Y["gas"].max()
tot = 0.0
for m in ("gas", "eui"):
    # P rows of the full OOF, S1 with the full-OOF deltas first (as in the candidate; those rows are far from the cap
    # anyway), then S3 with the candidate k
    p = PF[m].copy()
    x = s3_x(m, p, Pi, vmP[m], GMAX_ALL)
    q = p.copy()
    q[Pi] = p[Pi] + K_CAND[m] * x
    d = nr(m, q, Pi) - nr(m, p, Pi)
    tot += d / 5
    # best k on these rows (in-sample, reference only)
    kbest = float(x @ (Y[m][Pi] - p[Pi]) / (x @ x))
    print(f"   S3 candidate k {K_CAND[m]} -> full-OOF P rows {m:4s}: d {d:+.7f} (in-sample best k on P rows {kbest:.4f})")
print(f"   S3 regime transfer TOTAL on P rows {tot:+.8f}   [holdout chain S3 step: -0.00001923]")
summary["a_S3_fullOOF_Prows"] = tot

# (b)+(c) fully cross-fitted chain on the full OOF
res_tot = []
for seed in range(5):
    pred = {m: np.empty(n) for m in MET}
    ks = {"gas": [], "eui": []}
    for a, b in KFold(10, shuffle=True, random_state=100 + seed).split(A):
        gmax = Y["gas"][a].max()
        euimin = Y["eui"][a].min()
        ea = Y["elec"][a] / A[a]
        for m in MET:
            p = apply_s1(m, PF[m], b, deltas(m, a))
            pa = apply_s1(m, PF[m], a, deltas(m, a))  # fit rows get the same S1 before fitting k
            if m in ("gas", "eui"):
                vm = var_model(m, a)
                xa = s3_x(m, pa, a, vm, gmax)
                k = float(xa @ (Y[m][a] - pa[a]) / (xa @ xa))
                ks[m].append(k)
                p[b] = p[b] + k * s3_x(m, p, b, vm, gmax)
            lo = {"gas": euimin * A[b], "eui": np.full(len(b), euimin), "elec": ea.min() * A[b]}[m]
            hi = {"gas": np.full(len(b), gmax), "eui": gmax / A[b], "elec": ea.max() * A[b]}[m]
            pred[m][b] = np.clip(p[b], lo, hi)
    allr = np.arange(n)
    dm = {m: nr(m, pred[m], allr) - nr(m, PF[m], allr) for m in MET}
    t = sum(dm.values()) / 5
    res_tot.append(t)
    print(f"(c) seed {seed}: " + " ".join(f"{m} {dm[m]:+.7f}" for m in MET) + f" TOTAL {t:+.8f}; k gas "
          f"{np.round(np.mean(ks['gas']), 4)} eui {np.round(np.mean(ks['eui']), 4)}", flush=True)
    if seed == 0:
        # per-step decomposition for seed 0 (S1 only; S1+S3) with the same folds
        pass
print(f"(c) full-OOF cross-fitted chain TOTAL mean {np.mean(res_tot):+.8f} (min {np.min(res_tot):+.8f} max "
      f"{np.max(res_tot):+.8f})   [holdout chain claim: -0.0000652]")
summary["c_fullOOF_crossfit_total"] = res_tot

# step decomposition on the full OOF (seed 100 folds): S1 alone, S3 alone
for steps in (("S1",), ("S3",), ("S4",)):
    pred = {m: PF[m].copy() for m in MET}
    for a, b in KFold(10, shuffle=True, random_state=100).split(A):
        gmax, euimin, ea = Y["gas"][a].max(), Y["eui"][a].min(), Y["elec"][a] / A[a]
        for m in MET:
            p = PF[m].copy()
            if "S1" in steps:
                p = apply_s1(m, p, b, deltas(m, a))
            if "S3" in steps and m in ("gas", "eui"):
                vm = var_model(m, a)
                xa = s3_x(m, PF[m], a, vm, gmax)
                k = float(xa @ (Y[m][a] - PF[m][a]) / (xa @ xa))
                p[b] = p[b] + k * s3_x(m, p, b, vm, gmax)
            if "S4" in steps:
                lo = {"gas": euimin * A[b], "eui": np.full(len(b), euimin), "elec": ea.min() * A[b]}[m]
                hi = {"gas": np.full(len(b), gmax), "eui": gmax / A[b], "elec": ea.max() * A[b]}[m]
                p[b] = np.clip(p[b], lo, hi)
            pred[m][b] = p[b]
    allr = np.arange(n)
    dm = {m: nr(m, pred[m], allr) - nr(m, PF[m], allr) for m in MET}
    print(f"   full-OOF step {'+'.join(steps)} alone: " + " ".join(f"{m} {dm[m]:+.7f}" for m in MET)
          + f" TOTAL {sum(dm.values()) / 5:+.8f}")
    summary[f"c_step_{'+'.join(steps)}"] = sum(dm.values()) / 5

# bootstrap over train rows of the cross-fitted chain (seed 0 folds) -> sampling sd at 429k rows
pred = {m: np.empty(n) for m in MET}
for a, b in KFold(10, shuffle=True, random_state=100).split(A):
    gmax, euimin, ea = Y["gas"][a].max(), Y["eui"][a].min(), Y["elec"][a] / A[a]
    for m in MET:
        p = apply_s1(m, PF[m], b, deltas(m, a))
        pa = apply_s1(m, PF[m], a, deltas(m, a))
        if m in ("gas", "eui"):
            vm = var_model(m, a)
            xa = s3_x(m, pa, a, vm, gmax)
            k = float(xa @ (Y[m][a] - pa[a]) / (xa @ xa))
            p[b] = p[b] + k * s3_x(m, p, b, vm, gmax)
        lo = {"gas": euimin * A[b], "eui": np.full(len(b), euimin), "elec": ea.min() * A[b]}[m]
        hi = {"gas": np.full(len(b), gmax), "eui": gmax / A[b], "elec": ea.max() * A[b]}[m]
        pred[m][b] = np.clip(p[b], lo, hi)
e0 = {m: (Y[m] - PF[m]) ** 2 for m in MET}
e1 = {m: (Y[m] - pred[m]) ** 2 for m in MET}
rng = np.random.default_rng(7)
bt = []
for i in range(500):
    idx = rng.integers(0, n, n)
    bt.append(sum(np.sqrt(e1[m][idx].mean()) / SIG[m] - np.sqrt(e0[m][idx].mean()) / SIG[m] for m in MET) / 5)
bt = np.array(bt)
print(f"   bootstrap full OOF (500x): mean {bt.mean():+.8f} sd {bt.std():.8f} frac>=0 {np.mean(bt >= 0):.3f}")
summary["c_bootstrap"] = {"mean": float(bt.mean()), "sd": float(bt.std()), "frac_ge0": float(np.mean(bt >= 0))}

# (d) seen / unseen subzone mix for small-M rows
cnt = tr["subzone_id"].value_counts()
seen_te = te["subzone_id"].map(cnt).fillna(0).values > 0
single_tr = tr["subzone_id"].map(cnt).values == 1  # OOF analogue of 'unseen': the row is alone in its subzone
Mt = te["bld_dim_12"].values
for name, g, gt in (("M2-3", (M >= 2) & (M <= 3), (Mt >= 2) & (Mt <= 3)), ("M4", M == 4, Mt == 4), ("all", M > 0, Mt > 0)):
    line = f"(d) {name:4s}: test unseen share {np.mean(~seen_te[gt]):.3f} (n {int(gt.sum())}) | train alone-in-subzone share {np.mean(single_tr[g]):.3f} (n {int(g.sum())})"
    for m in MET:
        for lab, gg in (("alone", g & single_tr), ("shared", g & ~single_tr)):
            if gg.sum() > 5:
                line += f" | {m} {lab} bias {(Y[m][gg] - PF[m][gg]).sum() / PF[m][gg].sum():+.3f} (n {int(gg.sum())})"
    print(line)
json.dump(summary, open(f"{OUT}/c03_transfer.json", "w"), indent=1, default=float)
