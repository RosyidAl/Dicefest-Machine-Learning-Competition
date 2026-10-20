"""verify_postproc/adv v03: winner's-curse check. The recommended chain (S1 M<=3 + M=4 multiplicative, S3, S4, no S2)
was picked after comparing variants on H. Here the whole selection is nested: on half A of H every variant gets a
5-fold cross-fitted score (as the auditor scored variants), the best one is selected, then it is scored on half B with
its H-dependent parameters fitted on all of A. 20 random halvings x 2 directions.

Variant space (what the auditor's logs show was compared on H):
  S1: none | M<=3 mult | M<=3 + M=4 mult | M<=3 + M4-5 pooled mult | M<=3 + M=4 additive  (deltas from P-OOF only)
  S2: off | gas-EUI consistency with hat(rank A, 2 knots) weights fitted on H fit rows
  S3: off | truncated-normal soft cap (gas, eui), k fitted on H fit rows
  S4: off | hard clip to P bounds
Reports: mean A-score of the selected variant (optimistic), its B-score (honest for the selection procedure), and the
fixed recommended chain on the same halves; optimism = A_sel - B_sel.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/adv/v03_nested_selection.py
"""
import itertools
import json
import sys

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[:3]
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/adv"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
nH = len(H)
y = {m: z[f"{m}_y"] for m in MET}
pP = {m: z[f"{m}_pP"] for m in MET}
rP = {m: z[f"{m}_rP"] for m in MET}
yP = {m: pP[m] + rP[m] for m in MET}
base = {m: np.maximum(z[f"{m}_corrected"], 0.0) for m in MET}
gmax, euimin = yP["gas"].max(), yP["eui"].min()
ea = yP["elec"] / AP
LOHI = {"gas": (euimin * AH, np.full(nH, gmax)), "eui": (np.full(nH, euimin), gmax / AH),
        "elec": (ea.min() * AH, ea.max() * AH)}
VAR = {m: np.linalg.lstsq(np.column_stack([np.ones(len(P)), 1.0 / MP]), (rP[m] / pP[m]) ** 2, rcond=None)[0]
       for m in ("gas", "eui")}
rankA = (np.argsort(np.argsort(AH, kind="stable"), kind="stable") + 0.5) / nH
HAT = np.column_stack([1 - rankA, rankA])
S1OPT = {"none": (), "M3": ((2, 3),), "M3+M4": ((2, 3), (4, 4)), "M3+M45": ((2, 3), (4, 5)), "M3+M4add": ((2, 3), (4, 4))}


def s1(form):
    """S1 output for all H rows (no H labels used)."""
    out = {}
    for m in MET:
        q = base[m].copy()
        for a, b in S1OPT[form]:
            gp, gh = (MP >= a) & (MP <= b), (MH >= a) & (MH <= b)
            if form.endswith("add"):
                q[gh] = q[gh] + float(rP[m][gp].mean())
            else:
                q[gh] = q[gh] * (1.0 + float(pP[m][gp] @ rP[m][gp] / (pP[m][gp] @ pP[m][gp])))
        out[m] = q
    return out


S1PRE = {f: s1(f) for f in S1OPT}


def tshift(p, s, U):
    b = (U - p) / s
    return -s * np.exp(norm.logpdf(b) - log_ndtr(b))


def apply(variant, fit):
    """full-H predictions of a variant; H-dependent parameters fitted on rows `fit` only"""
    f1, s2, s3, s4 = variant
    cur = {m: S1PRE[f1][m].copy() for m in MET}
    if s2:
        g, e = cur["gas"].copy(), cur["eui"].copy()
        for m, p, other in (("gas", g, e * AH), ("eui", e, g / AH)):
            X = HAT * (other - p)[:, None]
            beta = np.linalg.lstsq(X[fit], (y[m] - p)[fit], rcond=None)[0]
            cur[m] = np.maximum(p + X @ beta, 0.0)
    if s3:
        for m in ("gas", "eui"):
            p = cur[m]
            x = tshift(p, p * np.sqrt(np.maximum(VAR[m][0] + VAR[m][1] / MH, 1e-8)), LOHI[m][1])
            cur[m] = p + float(x[fit] @ (y[m] - p)[fit] / (x[fit] @ x[fit])) * x
    if s4:
        for m in MET:
            cur[m] = np.clip(cur[m], *LOHI[m])
    return cur


def total(pred, idx):
    """change of the TOTAL score on rows idx (3 changed metrics, divided by 5) vs the base"""
    d = 0.0
    for m in MET:
        d += (np.sqrt(np.mean((y[m][idx] - pred[m][idx]) ** 2))
              - np.sqrt(np.mean((y[m][idx] - base[m][idx]) ** 2))) / SIGMAS[COL[m]]
    return float(d / 5)


VARIANTS = [v for v in itertools.product(S1OPT, (False, True), (False, True), (False, True))]
FIXED = ("M3+M4", False, True, True)
needs_fit = lambda v: v[1] or v[2]  # noqa: E731


def cv_score(variant, rows, seed=0):
    """5-fold cross-fitted score of a variant inside `rows` (the auditor's way of scoring a variant)"""
    if not needs_fit(variant):
        return total(apply(variant, rows), rows)
    pred = {m: np.empty(nH) for m in MET}
    for a, b in KFold(5, shuffle=True, random_state=seed).split(rows):
        pr = apply(variant, rows[a])
        for m in MET:
            pred[m][rows[b]] = pr[m][rows[b]]
    return total(pred, rows)


# full-H cross-fitted scores of every variant (what the auditor could see)
full = {v: cv_score(v, np.arange(nH)) for v in VARIANTS}
rank = sorted(full, key=full.get)
print("full-H 5-fold scores, best 8 variants:")
for v in rank[:8]:
    print(f"   {str(v):38s} {full[v]:+.8f}")
print(f"   fixed {FIXED} {full[FIXED]:+.8f} (rank {rank.index(FIXED) + 1} of {len(VARIANTS)})")

rows = []
for rep in range(20):
    perm = np.random.default_rng(900 + rep).permutation(nH)
    h1, h2 = np.sort(perm[: nH // 2]), np.sort(perm[nH // 2:])
    for a_, b_ in ((h1, h2), (h2, h1)):
        sc = {v: cv_score(v, a_) for v in VARIANTS}
        sel = min(sc, key=sc.get)
        b_sel = total(apply(sel, a_), b_)
        b_fix = total(apply(FIXED, a_), b_)
        rows.append(dict(sel=str(sel), a_sel=sc[sel], b_sel=b_sel, a_fix=sc[FIXED], b_fix=b_fix))
R = {k: np.array([r[k] for r in rows]) for k in ("a_sel", "b_sel", "a_fix", "b_fix")}
sel_counts = {}
for r in rows:
    sel_counts[r["sel"]] = sel_counts.get(r["sel"], 0) + 1
print(f"\nnested over {len(rows)} half-splits:")
print(f"   selected variant: A-score mean {R['a_sel'].mean():+.8f} | B-score mean {R['b_sel'].mean():+.8f} "
      f"(sd {R['b_sel'].std():.8f}, frac<0 {np.mean(R['b_sel'] < 0):.2f}) | optimism A-B {np.mean(R['a_sel'] - R['b_sel']):+.8f}")
print(f"   fixed chain     : A-score mean {R['a_fix'].mean():+.8f} | B-score mean {R['b_fix'].mean():+.8f} "
      f"(sd {R['b_fix'].std():.8f}, frac<0 {np.mean(R['b_fix'] < 0):.2f}) | A-B {np.mean(R['a_fix'] - R['b_fix']):+.8f}")
print("   selection counts", dict(sorted(sel_counts.items(), key=lambda t: -t[1])))
json.dump(dict(full={str(k): v for k, v in full.items()}, nested=rows, fixed=str(FIXED)),
          open(f"{OUT}/v03_nested_selection.json", "w"), indent=1, default=float)
