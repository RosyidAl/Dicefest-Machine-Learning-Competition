"""verify_postproc v1: re-derive the S3 soft-cap gain on the honest H chain from scratch.

(a) S3 alone and S1 -> S3 with k cross-fitted 5-fold on H (several fold seeds, not only random_state 0).
(b) Fully nested version: the residual-correction lambda of the training folds is itself re-cross-fitted INSIDE the
    training folds, so no label of the scored fold reaches the k fit even at second order (in chain_H the corrected
    predictions of training-fold rows were built with a lambda that saw the scored fold).
(c) repeated split-half (fit k on one half, score the other half).
(d) residual bias near the cap for teacher vs corrected (does the correction itself change the near-cap bias?).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/v1_reproduce_H.py
"""
import sys

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
MP, MH, AH = Mall[P], Mall[H], A[H]
GMAX = float((z["gas_pP"] + z["gas_rP"]).max())
T = {"gas": np.full(len(H), GMAX), "eui": GMAX / AH}
SD = {}
for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]
y = {m: z[f"{m}_y"] for m in SIG}
teacher = {m: z[f"{m}_teacher"] for m in SIG}
corr = {m: np.clip(z[f"{m}_corrected"], 0, None) for m in SIG}
print("GMAX", GMAX, "SD", {m: np.round(v, 5).tolist() for m, v in SD.items()})


def nr(m, idx, p):
    return float(np.sqrt(np.mean((y[m][idx] - p) ** 2)) / SIG[m])


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


def s1(m, p):
    p = p.copy()
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    return p


def xfeat(m, p):
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / MH, 1e-8, None))
    return tn_hi(p, s, T[m])


def kfit(x, r):
    return float(x @ r / (x @ x))


# (a) cross-fitted k, several fold seeds
print("== (a) cross-fitted S3 gain on H (5-fold), per fold seed")
for with_s1 in (False, True):
    for m in SIG:
        p = s1(m, corr[m]) if with_s1 else corr[m]
        x, r = xfeat(m, p), y[m] - p
        base = nr(m, slice(None), p)
        ds, ks_all = [], []
        for seed in range(10):
            pc = p.copy()
            ks = []
            for a, b in KFold(5, shuffle=True, random_state=seed).split(p):
                k = kfit(x[a], r[a])
                ks.append(k)
                pc[b] = p[b] + k * x[b]
            ds.append(nr(m, slice(None), pc) - base)
            ks_all.append(np.round(ks, 3).tolist())
        print(f"   {'S1->' if with_s1 else ''}S3 {m}: base {base:.6f} d(seed0) {ds[0]:+.7f} d mean over 10 seeds "
              f"{np.mean(ds):+.7f} (min {np.min(ds):+.7f} max {np.max(ds):+.7f}) total {np.mean(ds) / 5:+.8f};"
              f" k seed0 {ks_all[0]} full-H k {kfit(x, r):.4f}")

# (b) fully nested: correction lambda re-cross-fitted inside the training folds
print("== (b) fully nested (correction lambda refit inside training folds), S1 -> S3")
for m in SIG:
    F = z[f"{m}_F"].astype(np.float64)
    rH = y[m] - teacher[m]
    # sanity: rebuilding the chain's corrected prediction from F reproduces z[corrected]
    pc0 = np.zeros(len(H))
    for a, b in KFold(5, shuffle=True, random_state=0).split(F):
        pc0[b] = F[b] @ np.linalg.lstsq(F[a], rH[a], rcond=None)[0]
    rebuilt = np.clip(teacher[m] + pc0, 0, None)
    print(f"   {m}: rebuild max |diff|/sigma {np.max(np.abs(rebuilt - corr[m])) / SIG[m]:.2e}")
    for seed in (0, 1, 2):
        out_base, out_s3 = np.empty(len(H)), np.empty(len(H))
        ks = []
        for a, b in KFold(5, shuffle=True, random_state=seed).split(F):
            lam_a = np.linalg.lstsq(F[a], rH[a], rcond=None)[0]
            pb_full = s1(m, np.clip(teacher[m] + F @ lam_a, 0, None))  # scored rows b: lambda from training folds only
            # training rows a: inner cross-fitted correction using only rows of a
            pa = teacher[m] + F @ lam_a  # placeholder for rows outside a (never used)
            Fa, ra, ta = F[a], rH[a], teacher[m][a]
            inner = np.empty(len(a))
            for ia, ib in KFold(5, shuffle=True, random_state=100 + seed).split(Fa):
                inner[ib] = ta[ib] + Fa[ib] @ np.linalg.lstsq(Fa[ia], ra[ia], rcond=None)[0]
            pa[a] = inner
            pa_full = s1(m, np.clip(pa, 0, None))
            k = kfit(xfeat(m, pa_full)[a], y[m][a] - pa_full[a])
            ks.append(k)
            xb = xfeat(m, pb_full)[b]
            out_base[b] = pb_full[b]
            out_s3[b] = pb_full[b] + k * xb
        d = nr(m, slice(None), out_s3) - nr(m, slice(None), out_base)
        print(f"   {m} seed {seed}: base {nr(m, slice(None), out_base):.6f} d {d:+.7f} total {d / 5:+.8f} k {np.round(ks, 3).tolist()}")

# (c) repeated split-half, S1 -> S3, both directions
print("== (c) split-half S1 -> S3 (40 = 20 splits x 2 directions)")
tot = []
for rep in range(20):
    perm = np.random.default_rng(900 + rep).permutation(len(H))
    h1, h2 = np.sort(perm[: len(H) // 2]), np.sort(perm[len(H) // 2:])
    for a, b in ((h1, h2), (h2, h1)):
        t = 0.0
        for m in SIG:
            p = s1(m, corr[m])
            x = xfeat(m, p)
            k = kfit(x[a], (y[m] - p)[a])
            t += (nr(m, b, p[b] + k * x[b]) - nr(m, b, p[b])) / 5
        tot.append(t)
tot = np.array(tot)
print(f"   total mean {tot.mean():+.8f} sd {tot.std():.8f} frac<0 {np.mean(tot < 0):.3f}  (x2 for full-H scale is wrong; "
      f"half-sample deltas are on half the rows, the mean is comparable to the full-H estimate)")

# (d) bias near the cap: teacher vs corrected (binned by standardized distance using each one's own p)
print("== (d) near-cap residual means on H: teacher vs corrected")
for m in SIG:
    for name, p in (("teacher", teacher[m]), ("corrected", corr[m])):
        s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / MH, 1e-8, None))
        dist = (T[m] - p) / s
        row = []
        for lo, hi in ((-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3)):
            sel = (dist >= lo) & (dist < hi)
            r = (y[m] - p)[sel]
            row.append(f"[{lo},{hi}) n{sel.sum():4d} {r.mean() / SIG[m]:+.3f}s(t{r.mean() / (r.std() / np.sqrt(max(sel.sum(), 2)) + 1e-12):+.1f})")
        print(f"   {m:4s} {name:9s} " + " ".join(row))
