"""corrhyper c02: how the correction weights lambda are fitted (same features as the chain, exp/audit/chain_H.npz).

Variants of the lstsq on H (all 5-fold cross-fitted on H, same folds as the chain; split-half check over 4 splits):
  ols     current (np.linalg.lstsq, no intercept)
  icpt    + intercept column
  nnls    lambda >= 0 (scipy.optimize.nnls)
  ridge a ridge with penalty a * trace(F'F)/p on standardized-scale lambda (a in 1e-4..1e-1)
  wls     weighted lstsq, weight 1/variance proxy (teacher^2 * (a + b/M), noise model from memory; EUI: (a + b/M) / ...)
Also: does KSHR matter for TEST rows? pool counts with all train rows as fit rows (c = min(k, pool)).
"""
import json

import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
NOISE = {"gas": (0.0091, 0.377), "eui": (0.0091, 0.377), "elec": (0.0113, 0.591)}
CFG = json.load(open("exp/v19vce/resid_knn.json"))["configs"]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M, subn = z["P"], z["H"], z["A"], z["M"], z["subn"]
HALVES = [np.random.default_rng(s).permutation(len(H)) % 2 for s in (11, 12, 13, 14)]
logf = open("exp/audit/corrhyper/c02_lamfit.log", "w")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def solve(kind, Fa, ra, wa):
    if kind == "ols":
        return np.linalg.lstsq(Fa, ra, rcond=None)[0]
    if kind == "nnls":
        s = np.abs(Fa).mean(0) + 1e-30
        return nnls(Fa / s, ra, maxiter=2000)[0] / s
    if kind.startswith("ridge"):
        a = float(kind[5:])
        s = np.sqrt((Fa ** 2).mean(0)) + 1e-30
        X = Fa / s
        G = X.T @ X
        return np.linalg.solve(G + a * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), X.T @ ra) / s
    if kind == "wls":
        sw = np.sqrt(wa)
        return np.linalg.lstsq(Fa * sw[:, None], ra * sw, rcond=None)[0]
    raise ValueError(kind)


def cv(kind, F, r, w, sig, idx=None):
    if idx is not None:
        F, r, w = F[idx], r[idx], w[idx]
    pc = np.zeros(len(r))
    for a, b in KFold(5, shuffle=True, random_state=0).split(F):
        pc[b] = F[b] @ solve(kind, F[a], r[a], w[a])
    return np.sqrt(np.mean((r - pc) ** 2)) / sig


KINDS = ["ols", "icpt", "nnls", "ridge0.0001", "ridge0.001", "ridge0.01", "ridge0.1", "wls"]
OUT, tot_h, tot_a = {}, 0.0, 0.0
for m in ("gas", "eui", "elec"):
    F = z[f"{m}_F"].astype(np.float64)
    pH = z[f"{m}_teacher"]
    rH = z[f"{m}_y"] - pH
    a0, b0 = NOISE[m]
    w = 1.0 / (np.maximum(pH, 1e-9) ** 2 * (a0 + b0 / M[H]))
    if m == "eui":  # EUI noise variance ~ 1/area as well (household averaging); keep the same form on the EUI scale
        w = 1.0 / (np.maximum(pH, 1e-9) ** 2 * (a0 + b0 / M[H]))
    sig = SIG[m]
    full, half = {}, {}
    for k in KINDS:
        Fk = np.column_stack([F, np.ones(len(F))]) if k == "icpt" else F
        kk = "ols" if k == "icpt" else k
        full[k] = cv(kk, Fk, rH, w, sig)
        half[k] = [[cv(kk, Fk, rH, w, sig, hv == h) for h in (0, 1)] for hv in HALVES]
    log(f"== {m}: ols {full['ols']:.6f} (stored corrected "
        f"{np.sqrt(np.mean((z[f'{m}_y'] - z[f'{m}_corrected']) ** 2)) / sig:.6f})")
    for k in KINDS[1:]:
        d = [half[k][si][h] - half["ols"][si][h] for si in range(4) for h in (0, 1)]
        log(f"  {k:12s} all-H {full[k] - full['ols']:+.7f}; per-half diffs mean {np.mean(d):+.7f} frac<0 {np.mean(np.array(d) < 0):.2f}")
    hon = []
    for si in range(4):
        for h in (0, 1):
            g = min(KINDS, key=lambda k: half[k][si][h])
            hon.append(half[g][si][1 - h] - half["ols"][si][1 - h])
    best = min(KINDS, key=full.get)
    log(f"  split-half honest (choose among {len(KINDS)} kinds on one half, score other): {np.mean(hon):+.7f}; "
        f"best on all H {best} {full[best] - full['ols']:+.7f}")
    OUT[m] = dict(full={k: float(v - full["ols"]) for k, v in full.items()}, honest=float(np.mean(hon)))
    tot_h += np.mean(hon) / 5
    tot_a += (full[best] - full["ols"]) / 5

# KSHR at test time: fit rows = all train rows, apply rows = test rows. Test sub_n are not in chain_H; train rows are
# a 70% random sample of all rows, so use H rows against all train rows minus themselves as the denser analogue.
allt = np.sort(subn)
for m in ("gas", "eui", "elec"):
    short = []
    for c in CFG[m]:
        W, k = c[1], c[2]
        pool = np.searchsorted(allt, subn[H] + W, "right") - np.searchsorted(allt, subn[H] - W, "left") - 1
        short.append(float(np.mean(pool < k)))
    log(f"KSHR test analogue {m}: share of rows with pool < k per config {short}")
log(f"TOTAL lambda-fit variants: honest split-half {tot_h:+.8f}; all-H best (optimistic) {tot_a:+.8f}")
OUT["total"] = dict(honest=tot_h, allH=tot_a)
json.dump(OUT, open("exp/audit/corrhyper/c02_lamfit.json", "w"), indent=1)
