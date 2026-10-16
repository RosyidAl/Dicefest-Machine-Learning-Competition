"""Holdout blend check: blend weights fitted on OOF predictions (current practice) vs weights fitted on
fold-averaged holdout predictions, which behave like test predictions.

P = 70% of train (10-fold OOF in exp/hold/oof), H = the other 30% (fold-averaged predictions in exp/hold/pred).
usage: python src/eval_hold.py
"""
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from blend import cv_blend_score, nnls_blend  # noqa: E402
from common import SIGMAS, TARGET_COLS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402

tr, _ = load()
P, H = holdout_split(len(tr))
A = tr["bld_dim_06"].values
ONE = np.ones(len(A))
y = {c: tr[c].values for c in TARGET_COLS}
GAS, ELE, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]

runs = sorted(f[:-4] for f in os.listdir("exp/hold/pred") if os.path.getsize(f"exp/hold/pred/{f}") > 0)
el = [r for r in runs if any(t in r.split("_") for t in ELEC)]
gas = [r for r in runs if r not in el]
# (metric, member list, target in the stored scale, row scale, scored column)
METRICS = [("gas", gas, y[EUI], A, GAS), ("elec", el, y[ELE], ONE, ELE), ("eui", gas, y[EUI], ONE, EUI)]


def sq(yt, p, s):
    return (s * (yt - p)) ** 2


def nr(e, col):
    return np.sqrt(e.mean()) / SIGMAS[col]


errs = {}  # method -> list of per-row squared errors on H, one per metric
for m, members, yy, sc, col in METRICS:
    Po = np.column_stack([np.load(f"exp/hold/oof/{r}.npy") for r in members])
    Ph = np.column_stack([np.load(f"exp/hold/pred/{r}.npy") for r in members])
    yP, yH, sP, sH = yy[P], yy[H], sc[P], sc[H]
    w_oof = nnls_blend(Po, yP, sP)
    w_hold = nnls_blend(Ph, yH, sH)
    cvH = np.zeros(len(H))
    for a, b in KFold(5, shuffle=True, random_state=0).split(Ph):
        cvH[b] = Ph[b] @ nnls_blend(Ph[a], yH[a], sH[a])
    single = [nr(sq(yH, Ph[:, j], sH), col) for j in range(len(members))]
    jb = int(np.argmin(single))
    e = {"oof_w": sq(yH, Ph @ w_oof, sH), "hold_w_cv": sq(yH, cvH, sH), "hold_w_insample": sq(yH, Ph @ w_hold, sH),
         "best_single": sq(yH, Ph[:, jb], sH)}
    for k, v in e.items():
        errs.setdefault(k, []).append(v)
    cvP = nr(sq(yP, cv_blend_score(Po, yP, sP), sP), col)
    print(f"\n== {m}: {len(members)} members, H rows {len(H)}")
    print(f"  claimed CV on P (OOF blend, cross-fitted) {cvP:.5f}")
    for k, v in e.items():
        print(f"  H score {k:16s} {nr(v, col):.5f}")
    print(f"  {'member':16s} {'OOF(P)':>8s} {'H':>8s} {'H-OOF':>8s} {'w_oof':>6s} {'w_hold':>6s}")
    for j in np.argsort(single):
        o = nr(sq(yP, Po[:, j], sP), col)
        print(f"  {members[j]:16s} {o:8.5f} {single[j]:8.5f} {single[j] - o:+8.5f} {w_oof[j]:6.3f} {w_hold[j]:6.3f}")

print("\n== total over gas+elec+eui (/5, clean=0 and adoption 0.03565 added)")
tot = {k: (sum(nr(e, c) for e, (_, _, _, _, c) in zip(v, METRICS)) + 0.03565) / 5 for k, v in errs.items()}
for k, v in tot.items():
    print(f"  {k:16s} {v:.5f}")
rng = np.random.default_rng(0)
diffs = []
for _ in range(300):
    i = rng.integers(0, len(H), len(H))
    t = {k: sum(nr(e[i], c) for e, (_, _, _, _, c) in zip(v, METRICS)) / 5 for k, v in errs.items()
         if k in ("oof_w", "hold_w_cv")}
    diffs.append(t["hold_w_cv"] - t["oof_w"])
print(f"  hold_w_cv - oof_w: {tot['hold_w_cv'] - tot['oof_w']:+.5f} (bootstrap sd {np.std(diffs):.5f})")
