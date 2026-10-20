"""Audit postproc 11: the candidate post-processing chain on the honest H predictions, step by step.

S0 base   clip(corrected, 0)                                     (= current pipeline)
S1 smallM p *= 1 + delta_g for M <= 3 (and variant + M == 4), delta = least squares on P-OOF teacher residuals
S2 cons   gas' = gas + w(rankA) (eui*A - gas), eui' = eui + v(rankA) (gas/A - eui), hat basis 2 knots on rank A
          (weights fitted on H training folds)
S3 softcap p += k * tn_hi(p, s), s = p sqrt(a0 + b0/M) from P-OOF, k fitted on H training folds (gas, eui)
S4 hard   clip to the train-implied bounds (from P rows): gas <= max gas, eui <= max gas / A, eui >= min eui,
          5.13 A <= elec <= 149.99 A
Evaluation: 5-fold cross-fitting on H (all H-fitted parameters fitted on 4/5, applied to 1/5), and 20 repeated
split-halves (fit on one half, score on the other, both directions) for the spread of the total change.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/11_combo.py
"""
import sys

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
MET = ("gas", "eui", "elec")
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], Mall[P], Mall[H]
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in MET}
GMAX = yP["gas"].max()
EAMAX, EAMIN, EUIMIN = (yP["elec"] / AP).max(), (yP["elec"] / AP).min(), yP["eui"].min()
y = {m: z[f"{m}_y"] for m in MET}
base = {m: np.clip(z[f"{m}_corrected"], 0, None) for m in MET}
rankA = (np.argsort(np.argsort(AH)) + 0.5) / len(AH)


def hat(x, k):
    t = np.linspace(0, 1, k)
    return np.column_stack([np.clip(1 - np.abs(x - tj) * (k - 1), 0, None) for tj in t])


HAT = hat(rankA, 2)
SD = {}
for m in MET:
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(p)), 1 / MP]), (r / p) ** 2, rcond=None)[0]


def smallM_deltas(groups):
    out = {}
    for m in MET:
        pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
        out[m] = [float(pP[g] @ rP[g] / (pP[g] @ pP[g])) for g in ((MP >= lo) & (MP <= hi) for lo, hi in groups)]
    return out


def bounds(m, idx):
    a = AH[idx]
    return {"gas": (EUIMIN * a, np.full(len(a), GMAX)), "eui": (np.full(len(a), EUIMIN), GMAX / a),
            "elec": (EAMIN * a, EAMAX * a)}[m]


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


def chain(fit, app, groups, steps):
    """fit H-dependent parameters on rows `fit`, return the post-processed predictions for rows `app` per step"""
    res = {}
    cur = {m: base[m].copy() for m in MET}  # full-H arrays; the fit rows get the same transformation in sequence
    res["S0"] = {m: cur[m][app].copy() for m in MET}
    if "S1" in steps:
        dl = smallM_deltas(groups)
        for m in MET:
            for (lo, hi), d in zip(groups, dl[m]):
                g = (MH >= lo) & (MH <= hi)
                cur[m][g] *= 1 + d
    res["S1"] = {m: cur[m][app].copy() for m in MET}
    if "S2" in steps:
        g, e = cur["gas"].copy(), cur["eui"].copy()
        for m, p, other in (("gas", g, e * AH), ("eui", e, g / AH)):
            X = HAT * (other - p)[:, None]
            beta = np.linalg.lstsq(X[fit], (y[m] - p)[fit], rcond=None)[0]
            cur[m] = np.clip(p + X @ beta, 0, None)
    res["S2"] = {m: cur[m][app].copy() for m in MET}
    if "S3" in steps:
        for m in ("gas", "eui"):
            p = cur[m]
            s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / MH, 1e-8, None))
            x = tn_hi(p, s, bounds(m, np.arange(len(H)))[1])
            k = float(x[fit] @ (y[m] - p)[fit] / (x[fit] @ x[fit]))
            cur[m] = p + k * x
    res["S3"] = {m: cur[m][app].copy() for m in MET}
    if "S4" in steps:
        for m in MET:
            lo, hi = bounds(m, np.arange(len(H)))
            cur[m] = np.clip(cur[m], lo, hi)
    res["S4"] = {m: cur[m][app].copy() for m in MET}
    return res


def nr(m, idx, p):
    return float(np.sqrt(np.mean((y[m][idx] - p) ** 2)) / SIG[m])


ALL = ("S1", "S2", "S3", "S4")
for gname, groups in (("M<=3", [(2, 3)]), ("M<=3,M=4", [(2, 3), (4, 4)])):
    print(f"######## small-M groups {gname}; P-OOF deltas {smallM_deltas(groups)}")
    # 5-fold cross-fitting: per fold, fit on 4/5, apply to 1/5, assemble full-H predictions per step
    preds = {s: {m: np.empty(len(H)) for m in MET} for s in ("S0",) + ALL}
    for a, b in KFold(5, shuffle=True, random_state=0).split(AH):
        r = chain(a, b, groups, ALL)
        for s in preds:
            for m in MET:
                preds[s][m][b] = r[s][m]
    idx = np.arange(len(H))
    prev = None
    for s in ("S0",) + ALL:
        sc = {m: nr(m, idx, preds[s][m]) for m in MET}
        if prev is None:
            print(f"   {s}: " + " ".join(f"{m} {sc[m]:.6f}" for m in MET))
            s0 = sc
        else:
            print(f"   {s}: " + " ".join(f"{m} {sc[m]:.6f} ({sc[m] - prev[m]:+.7f})" for m in MET)
                  + f" | step total {sum(sc[m] - prev[m] for m in MET) / 5:+.8f} cum total {sum(sc[m] - s0[m] for m in MET) / 5:+.8f}")
        prev = sc
    # repeated split-half for the full chain and for each single step alone
    for steps in (ALL, ("S1",), ("S2",), ("S3",), ("S4",), ("S1", "S3", "S4")):
        tots = []
        for rep in range(20):
            perm = np.random.default_rng(500 + rep).permutation(len(H))
            h1, h2 = np.sort(perm[: len(H) // 2]), np.sort(perm[len(H) // 2:])
            for a, b in ((h1, h2), (h2, h1)):
                r = chain(a, b, groups, steps)
                tots.append(sum(nr(m, b, r["S4"][m]) - nr(m, b, r["S0"][m]) for m in MET) / 5)
        tots = np.array(tots)
        print(f"   split-half steps {'+'.join(steps):12s}: total mean {tots.mean():+.8f} sd {tots.std():.8f} frac<0 {np.mean(tots < 0):.2f}")
