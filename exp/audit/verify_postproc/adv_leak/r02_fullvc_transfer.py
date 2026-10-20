"""Adversarial check of S1 on the full-train OOF and on the test side (leakage / optimism lens).

Independent re-implementation of the production varying-coefficient blend (spec A2xM2xp2, blend.blend_vc) with a
Gram / Cholesky NNLS (no 742 MB design matrix); validated against final/src/postprocess.blend_preds on elec.
  a. full-OOF LS deltas per group (the constants the action would ship) with HC0 / jackknife se; vs claimed values.
  b. test side: relative shift the residual correction already applies to small-M test rows (sub_v19vce - teacher),
     so that S1 would not double count; candidate file ratio check.
  c. expected test change: naive (-S d^2), debiased (S (2 se^2 - d^2)) and p-tercile-aware (bias differs by p level),
     using the test predictions of sub_v19vce.
  d. honest repeated 70/30 splits of the full OOF: deltas fitted on 70 %, gain scored on 30 % (private-LB-sized).
  e. specificity: bias of low-p rows with M >= 5 (is it M, or just a low-prediction shrinkage?).
  f. exchangeability of small-M test rows vs train rows (A/M, E/M, teacher p): Mann-Whitney / KS.
"""
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from scipy.stats import ks_2samp, mannwhitneyu

ROOT = "/root/Dicefest"
os.chdir(ROOT)
sys.path.insert(0, "src")
OUT = "exp/audit/verify_postproc/adv_leak"
t0 = time.time()
from common import TARGET_COLS  # noqa: E402

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
COL = {"gas": TARGET_COLS[0], "elec": TARGET_COLS[1], "eui": TARGET_COLS[2]}
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
CLAIM = {"gas": (0.19942, 0.06202), "eui": (0.18166, 0.03016), "elec": (0.18714, 0.07098)}
log = []
res = {}


def say(s=""):
    print(s, flush=True)
    log.append(s)


def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


tr = pd.read_parquet("cache/train.parquet", columns=sorted(set(KEY + TARGET_COLS)))
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
E = tr["bld_dim_01"].values.astype(float)
Et = te["bld_dim_01"].values.astype(float)
members = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
Pg = (np.column_stack([np.load(f"exp/oof/{r}.npy") for r in members["gas"]]),
      np.column_stack([np.load(f"exp/pred/{r}.npy") for r in members["gas"]]))
Pe = (np.column_stack([np.load(f"exp/oof/{r}.npy") for r in members["elec"]]),
      np.column_stack([np.load(f"exp/pred/{r}.npy") for r in members["elec"]]))
say(f"loaded: gas members {Pg[0].shape[1]}, elec members {Pe[0].shape[1]}; maxrss {mem()} MB")


def rank_mid(fit, app):
    s = np.sort(fit)
    return (np.searchsorted(s, app, side="left") + np.searchsorted(s, app, side="right")) / (2.0 * len(s))


def basis(vf, va):
    """product of 2-knot hat bases (knots 0, 1) on the mid-ranks of A, M, p within the fit rows: 8 columns"""
    B = np.ones((len(va["A"]), 1))
    for v in ("A", "M", "p"):
        t = rank_mid(vf[v], va[v])
        b2 = np.column_stack([1 - t, t])
        B = np.column_stack([B[:, i] * b2[:, j] for i in range(B.shape[1]) for j in range(2)])
    return B


def nnls_gram(Pf, B, y, s, chunk=25000):
    k, m = B.shape[1], Pf.shape[1]
    G = np.zeros((k * m, k * m))
    c = np.zeros(k * m)
    for a in range(0, len(y), chunk):
        b = slice(a, a + chunk)
        X = np.concatenate([Pf[b] * B[b, [j]] for j in range(k)], axis=1) * s[b, None]
        G += X.T @ X
        c += X.T @ (y[b] * s[b])
    jit = 0.0
    for _ in range(8):
        try:
            L = cholesky(G + jit * np.eye(len(G)) * np.mean(np.diag(G)), lower=True)
            break
        except np.linalg.LinAlgError:
            jit = 1e-14 if jit == 0 else jit * 10
    bvec = solve_triangular(L, c, lower=True)
    w, _ = nnls(L.T, bvec, maxiter=50 * len(c))
    return w.reshape(k, m).T, jit


def vc_blend(Pf, Pa, yf, Af, Aa, Mf, Ma, s):
    w, _ = nnls(Pf * s[:, None], yf * s)
    vf = dict(A=Af, M=Mf, p=Pf @ w)
    va = dict(A=Aa, M=Ma, p=Pa @ w)
    Bf, Ba = basis(vf, vf), basis(vf, va)
    W, jit = nnls_gram(Pf, Bf, yf, s)
    return ((Pf @ W) * Bf).sum(1), ((Pa @ W) * Ba).sum(1), jit


ones = np.ones(len(A))
yE, yG, yL = tr[COL["eui"]].values, tr[COL["gas"]].values, tr[COL["elec"]].values
pf, pa = {}, {}
for m, (P2, yf, s) in {"eui": (Pg, yE, ones), "gas": (Pg, yE, A), "elec": (Pe, yL, ones)}.items():
    f_, a_, jit = vc_blend(P2[0], P2[1], yf, A, At, M, Mt, s)
    pf[m], pa[m] = (f_ * A, a_ * At) if m == "gas" else (f_, a_)
    y = yG if m == "gas" else (yE if m == "eui" else yL)
    say(f"   VC refit {m:4s}: in-sample NRMSE {np.sqrt(np.mean((y - pf[m]) ** 2)) / SIG[m]:.5f} (jitter {jit}); "
        f"maxrss {mem()} MB")

# validation against the production function on elec (smallest design matrix)
sys.path.insert(0, "final/src")
from postprocess import blend_preds  # noqa: E402

f_prod, a_prod = blend_preds(Pe[0], Pe[1], yL, "elec", A, At, "A2xM2xp2", M, Mt)
say(f"   elec production blend_preds vs refit: max rel diff fit {np.max(np.abs(f_prod / pf['elec'] - 1)):.2e} "
    f"test {np.max(np.abs(a_prod / pa['elec'] - 1)):.2e}; maxrss {mem()} MB")
res["validation_elec_max_rel"] = float(max(np.max(np.abs(f_prod / pf['elec'] - 1)), np.max(np.abs(a_prod / pa['elec'] - 1))))
del f_prod, a_prod
Y = {"gas": yG, "eui": yE, "elec": yL}
G23, G4 = (M >= 2) & (M <= 3), M == 4
T23, T4 = (Mt >= 2) & (Mt <= 3), Mt == 4

# ---------------- a. full-OOF deltas ----------------
say("\n== a. full-OOF LS deltas (r = y - pf, metric scale)")
res["a"] = {}
D = {}
for m in SIG:
    r = Y[m] - pf[m]
    D[m] = []
    for g, s in (("M2-3", G23), ("M4", G4)):
        p_, r_ = pf[m][s], r[s]
        S = p_ @ p_
        d = (p_ @ r_) / S
        se = np.sqrt(np.sum((p_ * r_ - d * p_ ** 2) ** 2)) / S
        jk = np.array([(np.delete(p_, i) @ np.delete(r_, i)) / (np.delete(p_, i) @ np.delete(p_, i))
                       for i in range(len(p_))])
        se_jk = np.sqrt((len(jk) - 1) / len(jk) * np.sum((jk - jk.mean()) ** 2))
        D[m].append(d)
        claim = CLAIM[m][0 if g == "M2-3" else 1]
        res["a"][f"{m}_{g}"] = dict(n=int(s.sum()), d=d, se=se, se_jk=se_jk, claim=claim)
        say(f"   {m:4s} {g:4s} n {s.sum():4d} d {d:+.5f} (claim {claim:+.5f}, diff {d - claim:+.1e}) "
            f"se HC0 {se:.4f} jk {se_jk:.4f} t {d / se:+.1f}")

# ---------------- b. test side ----------------
say("\n== b. test side: what the residual correction already does for small-M rows; candidate ratios")
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
cand = pd.read_csv("exp/audit/postproc/sub_v19vce_pp.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all() and (cand["id"].values == te["id"].values).all()
from dup_fix import duplicate_targets  # noqa: E402

dup = ~np.isnan(duplicate_targets(tr, te)[:, 0])
say(f"   test duplicates {dup.sum()}, of which M<=4: {(dup & (Mt <= 4)).sum()}; test M=1 rows {(Mt == 1).sum()}")
res["b"] = {}
for m in SIG:
    fin = sub[COL[m]].values
    base = np.clip(pa[m], 0, None)
    corr = fin - base
    out = []
    for g, s in (("M2-3", T23 & ~dup), ("M4", T4 & ~dup), ("M>=5", (Mt >= 5) & ~dup)):
        out.append(f"{g} n {s.sum():6d} corr/teacher {corr[s].sum() / base[s].sum():+.4f}")
    ratio23 = cand[COL[m]].values[T23] / fin[T23] - 1
    ratio4 = cand[COL[m]].values[T4] / fin[T4] - 1
    say(f"   {m:4s} " + " | ".join(out) + f" || candidate/base - 1: M2-3 [{ratio23.min():.5f}, {ratio23.max():.5f}]"
        f" M4 [{ratio4.min():.5f}, {ratio4.max():.5f}]")
    res["b"][m] = dict(corr_rel_M23=float(corr[T23 & ~dup].sum() / base[T23 & ~dup].sum()),
                       corr_rel_M4=float(corr[T4 & ~dup].sum() / base[T4 & ~dup].sum()))

# ---------------- c. expected test change ----------------
say("\n== c. expected change on test (sub_v19vce predictions q of the touched rows, N = 183987)")
REF = {"gas": 0.24514, "eui": 0.64859, "elec": 0.31462}  # honest-analog NRMSE on H (corrected)
N = len(te)
res["c"] = {}
tot = {"naive": 0.0, "debiased": 0.0, "tercile": 0.0}
for m in SIG:
    q = sub[COL[m]].values
    r = Y[m] - pf[m]
    mse0 = (REF[m] * SIG[m]) ** 2
    dS = {"naive": 0.0, "debiased": 0.0, "tercile": 0.0}
    for gi, (gs, ts) in enumerate(((G23, T23 & ~dup), (G4, T4 & ~dup))):
        d = D[m][gi]
        se = res["a"][f"{m}_{'M2-3' if gi == 0 else 'M4'}"]["se"]
        S = q[ts] @ q[ts]
        dS["naive"] += -S * d * d
        dS["debiased"] += S * (2 * se * se - d * d)
        # p-tercile-specific relative bias b_k from train (LS within tercile of pf), applied to test rows by q tercile
        cuts = np.quantile(pf[m][gs], [1 / 3, 2 / 3])
        kt = np.digitize(pf[m][gs], cuts)
        ka = np.digitize(q[ts], cuts)
        for k in range(3):
            pk, rk = pf[m][gs][kt == k], r[gs][kt == k]
            bk = (pk @ rk) / (pk @ pk)
            Sk = q[ts][ka == k] @ q[ts][ka == k]
            dS["tercile"] += Sk * (d * d - 2 * d * bk)
    out = []
    for kk in dS:
        dn = (np.sqrt(mse0 + dS[kk] / N) - np.sqrt(mse0)) / SIG[m]
        tot[kk] += dn / 5
        out.append(f"{kk} {dn:+.7f}")
        res["c"][f"{m}_{kk}"] = float(dn)
    say(f"   {m:4s} expected dNRMSE: " + " | ".join(out))
say("   TOTAL expected: " + " | ".join(f"{k} {v:+.8f}" for k, v in tot.items()) + "   [claim analytic -0.0000434]")
res["c"]["total"] = tot

# ---------------- d. honest repeated 70/30 splits of the full OOF ----------------
say("\n== d. honest 70/30 splits of the full OOF (deltas on 70 %, gain on the 30 % = 128.8k rows), 400 repeats")
rng = np.random.default_rng(20261020)
n = len(A)
gains = []
per_m = {m: [] for m in SIG}
for rep in range(400):
    perm = rng.permutation(n)
    fit, ev = np.zeros(n, bool), np.zeros(n, bool)
    fit[perm[: int(0.7 * n)]] = True
    ev[perm[int(0.7 * n):]] = True
    g = 0.0
    for m in SIG:
        p_, y_ = pf[m], Y[m]
        r = y_ - p_
        q = p_.copy()
        for gs in (G23, G4):
            s = gs & fit
            d = (p_[s] @ r[s]) / (p_[s] @ p_[s])
            q[gs & ev] *= 1 + d
        b0 = np.sqrt(np.mean(r[ev] ** 2))
        b1 = np.sqrt(np.mean((y_[ev] - q[ev]) ** 2))
        per_m[m].append((b1 - b0) / SIG[m])
        g += (b1 - b0) / SIG[m] / 5
    gains.append(g)
gains = np.array(gains)
say(f"   TOTAL mean {gains.mean():+.8f} sd {gains.std():.8f} P(gain<0) {np.mean(gains < 0):.3f} "
    f"q05/50/95 {np.quantile(gains, [0.05, 0.5, 0.95])}")
for m in SIG:
    a = np.array(per_m[m])
    say(f"   {m:4s} dNRMSE mean {a.mean():+.7f} sd {a.std():.7f}")
res["d"] = dict(mean=float(gains.mean()), sd=float(gains.std()), p_neg=float(np.mean(gains < 0)))

# ---------------- e. specificity: low-prediction rows with M >= 5 ----------------
say("\n== e. specificity: LS relative bias of M>=5 rows in the lowest prediction bins (same p range as M2-3)")
for m in SIG:
    r = Y[m] - pf[m]
    lo, hi = np.quantile(pf[m][G23], [0.1, 0.9])
    s5 = (M >= 5) & (pf[m] >= lo) & (pf[m] <= hi)
    s5a = (M >= 5) & (M <= 6) & (pf[m] >= lo) & (pf[m] <= hi)
    out = []
    for nm, s in (("M>=5 in M2-3 p-range", s5), ("M5-6 in M2-3 p-range", s5a)):
        d = (pf[m][s] @ r[s]) / (pf[m][s] @ pf[m][s])
        se = np.sqrt(np.sum((pf[m][s] * r[s] - d * pf[m][s] ** 2) ** 2)) / (pf[m][s] @ pf[m][s])
        out.append(f"{nm}: n {s.sum()} d {d:+.4f} (t {d / se:+.1f})")
    say(f"   {m:4s} " + " | ".join(out))

# ---------------- f. exchangeability of small-M test rows ----------------
say("\n== f. small-M test rows vs train rows: Mann-Whitney / KS p-values")
for g, (gs, ts) in (("M2-3", (G23, T23)), ("M4", (G4, T4))):
    out = []
    for nm, a_, b_ in (("A/M", A[gs] / M[gs], At[ts] / Mt[ts]), ("E/M", E[gs] / M[gs], Et[ts] / Mt[ts]),
                       ("gas teacher", pf["gas"][gs], pa["gas"][ts]), ("eui teacher", pf["eui"][gs], pa["eui"][ts]),
                       ("elec teacher", pf["elec"][gs], pa["elec"][ts])):
        out.append(f"{nm} MW p {mannwhitneyu(a_, b_).pvalue:.3f} KS p {ks_2samp(a_, b_).pvalue:.3f} "
                   f"(median train {np.median(a_):.4g} test {np.median(b_):.4g})")
    say(f"   {g}: " + " | ".join(out))

say(f"\nmaxrss {mem()} MB wall {time.time() - t0:.1f} s")
np.savez_compressed(f"{OUT}/r02_vc_pred.npz", **{f"pf_{m}": pf[m] for m in SIG}, **{f"pa_{m}": pa[m] for m in SIG})
json.dump(res, open(f"{OUT}/r02_fullvc_transfer.json", "w"), indent=1, default=float)
open(f"{OUT}/r02_fullvc_transfer.log", "w").write("\n".join(log) + "\n")
