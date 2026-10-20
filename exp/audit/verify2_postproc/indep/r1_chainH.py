"""verify2_postproc/indep r1: independent re-implementation (own code) of the postproc chain on exp/audit/chain_H.npz.

S1  small-M multiplicative shift: p *= 1 + d_g for M in {2,3} and M == 4; d_g estimated on P-side teacher residuals
S3  soft upper cap (gas, eui): p += k * shift, shift = E[Y | Y < U] - p for Y ~ N(p, s^2), s = p sqrt(a0 + b0 / M)
    (a0, b0 from P residuals), U = max train gas (gas) or max gas / A (eui); k fitted by least squares
S4  hard clip to bounds implied by the P rows

Splits used here differ from the auditor's (they used KFold(5, seed 0) and split-halves seeds 500..519):
  - KFold(5) cross-fitting on H with seeds 1..60 (only S3's k uses H labels)
  - 200 split-halves with seeds 9000.., fit on one half and score on the other (both directions)
  - S3 with k fitted on the P side only (fully out-of-sample for H) and with k = 1 (no fit)
  - S1 reverse direction: deltas from H teacher residuals applied to the P-side OOF teacher, and 5-fold cross-fitting
    inside P
  - bootstrap of H rows (n = |H| and n = 55,196 = public-LB size) for the spread of the total change

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/indep/r1_chainH.py
"""
import json
import sys

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
MET = ["gas", "eui", "elec"]
OUT = "exp/audit/verify2_postproc/indep"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
nH = len(H)
y = {m: z[f"{m}_y"].astype(float) for m in MET}
base = {m: np.maximum(z[f"{m}_corrected"].astype(float), 0.0) for m in MET}
teach = {m: z[f"{m}_teacher"].astype(float) for m in MET}
pP = {m: z[f"{m}_pP"].astype(float) for m in MET}
rP = {m: z[f"{m}_rP"].astype(float) for m in MET}
yP = {m: pP[m] + rP[m] for m in MET}
GROUPS = [(2, 3), (4, 4)]
res = {}


def nrmse(m, idx, p):
    return float(np.sqrt(np.mean((y[m][idx] - p[idx]) ** 2)) / SIG[m])


def total_change(new, old, idx):
    return sum(nrmse(m, idx, new[m]) - nrmse(m, idx, old[m]) for m in MET) / 5


# ---------------------------------------------------------------- sanity: baseline and bounds
res["base_H"] = {m: nrmse(m, np.arange(nH), base[m]) for m in MET}
res["teacher_H"] = {m: nrmse(m, np.arange(nH), teach[m]) for m in MET}
bP = dict(gmax=float(yP["gas"].max()), euimin=float(yP["eui"].min()), eamin=float((yP["elec"] / AP).min()),
          eamax=float((yP["elec"] / AP).max()))
bH = dict(gmax=float(y["gas"].max()), euimin=float(y["eui"].min()), eamin=float((y["elec"] / AH).min()),
          eamax=float((y["elec"] / AH).max()))
res["bounds_P"], res["bounds_H_truth"] = bP, bH
# identity check gas == eui * A on both sides
res["identity_gas_eq_euiA_maxrel_H"] = float(np.max(np.abs(y["gas"] - y["eui"] * AH) / y["gas"]))
res["identity_gas_eq_euiA_maxrel_P"] = float(np.max(np.abs(yP["gas"] - yP["eui"] * AP) / yP["gas"]))
res["counts"] = {"H_M<=3": int((MH <= 3).sum()), "H_M=4": int((MH == 4).sum()), "P_M<=3": int((MP <= 3).sum()),
                 "P_M=4": int((MP == 4).sum()), "H_M_min": float(MH.min())}
print("base", res["base_H"], "teacher", res["teacher_H"], flush=True)
print("bounds P", bP, "| truth on H", bH, flush=True)
print("counts", res["counts"], flush=True)


# ---------------------------------------------------------------- S1
def deltas(p, r, Mv, est="ls"):
    out = []
    for lo, hi in GROUPS:
        g = (Mv >= lo) & (Mv <= hi)
        if est == "ls":
            out.append(float(p[g] @ r[g] / (p[g] @ p[g])))
        elif est == "ros":
            out.append(float(r[g].sum() / p[g].sum()))
        elif est == "mean":
            out.append(float(np.mean(r[g] / p[g])))
        elif est == "median":
            out.append(float(np.median(r[g] / p[g])))
    return out


def apply_s1(p, Mv, ds):
    q = p.copy()
    for (lo, hi), d in zip(GROUPS, ds):
        q[(Mv >= lo) & (Mv <= hi)] *= 1 + d
    return q


DP = {m: deltas(pP[m], rP[m], MP) for m in MET}
res["S1_deltas_P"] = DP
print("S1 deltas from P teacher residuals", DP, flush=True)
# bias of teacher vs corrected on H in the small-M groups (how much the correction already removed)
bias = {}
for m in MET:
    bias[m] = {}
    for lo, hi in GROUPS:
        g = (MH >= lo) & (MH <= hi)
        bias[m][f"M{lo}-{hi}"] = dict(n=int(g.sum()), teacher_ls=float(teach[m][g] @ (y[m][g] - teach[m][g]) / (teach[m][g] @ teach[m][g])),
                                      corrected_ls=float(base[m][g] @ (y[m][g] - base[m][g]) / (base[m][g] @ base[m][g])))
res["H_smallM_bias"] = bias
print("H small-M LS relative bias teacher vs corrected", json.dumps(bias), flush=True)

allH = np.arange(nH)
s1 = {}
for est in ("ls", "ros", "mean", "median"):
    ds = {m: deltas(pP[m], rP[m], MP, est) for m in MET}
    new = {m: apply_s1(base[m], MH, ds[m]) for m in MET}
    per = {m: nrmse(m, allH, new[m]) - nrmse(m, allH, base[m]) for m in MET}
    s1[est] = dict(per=per, total=sum(per.values()) / 5)
    print(f"S1 [{est:6s}] on H: " + " ".join(f"{m} {per[m]:+.7f}" for m in MET) + f" total {s1[est]['total']:+.8f}", flush=True)
# per-group contributions (ls)
for (lo, hi) in GROUPS:
    for m in MET:
        g = (MH >= lo) & (MH <= hi)
        q = base[m].copy()
        q[g] *= 1 + DP[m][GROUPS.index((lo, hi))]
        s1[f"ls_only_M{lo}-{hi}_{m}"] = nrmse(m, allH, q) - nrmse(m, allH, base[m])
res["S1_H"] = s1
print("S1 per group", {k: round(v, 8) for k, v in s1.items() if k.startswith("ls_only")}, flush=True)

# bootstrap of the S1 total change (no H fit involved, so the only uncertainty is the H sample)
new1 = {m: apply_s1(base[m], MH, DP[m]) for m in MET}
e_old = {m: (y[m] - base[m]) ** 2 for m in MET}
e_new = {m: (y[m] - new1[m]) ** 2 for m in MET}


def boot(e_new, e_old, n, reps, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        ix = rng.integers(0, nH, n)
        out[i] = sum(np.sqrt(e_new[m][ix].mean()) / SIG[m] - np.sqrt(e_old[m][ix].mean()) / SIG[m] for m in MET) / 5
    return out


bt = boot(e_new, e_old, nH, 2000, 11)
bp = boot(e_new, e_old, 55196, 2000, 12)
res["S1_boot_H"] = dict(mean=float(bt.mean()), sd=float(bt.std()), frac_neg=float((bt < 0).mean()),
                        q05=float(np.quantile(bt, .05)), q95=float(np.quantile(bt, .95)))
res["S1_boot_public_size"] = dict(mean=float(bp.mean()), sd=float(bp.std()), frac_neg=float((bp < 0).mean()))
print("S1 bootstrap |H|:", res["S1_boot_H"], "| public-size:", res["S1_boot_public_size"], flush=True)

# reverse direction: deltas from H teacher residuals -> P teacher (OOF in P); and 5-fold cross-fitting inside P
nP = len(P)


def nr_P(m, idx, p):
    return float(np.sqrt(np.mean((yP[m][idx] - p[idx]) ** 2)) / SIG[m])


DH = {m: deltas(teach[m], y[m] - teach[m], MH) for m in MET}
allP = np.arange(nP)
rev = {m: nr_P(m, allP, apply_s1(pP[m], MP, DH[m])) - nr_P(m, allP, pP[m]) for m in MET}
res["S1_reverse_H_to_P"] = dict(deltas_H=DH, per=rev, total=sum(rev.values()) / 5)
print("S1 reverse (H teacher deltas -> P teacher):", DH, {m: round(v, 8) for m, v in rev.items()},
      "total", round(sum(rev.values()) / 5, 8), flush=True)
cf = []
for seed in range(20):
    q = {m: pP[m].copy() for m in MET}
    for a, b in KFold(5, shuffle=True, random_state=seed).split(pP["gas"]):
        for m in MET:
            ds = deltas(pP[m][a], rP[m][a], MP[a])
            q[m][b] = apply_s1(pP[m][b], MP[b], ds)
    cf.append({m: nr_P(m, allP, q[m]) - nr_P(m, allP, pP[m]) for m in MET})
cft = np.array([sum(c.values()) / 5 for c in cf])
res["S1_crossfit_in_P"] = dict(per_mean={m: float(np.mean([c[m] for c in cf])) for m in MET}, total_mean=float(cft.mean()),
                               total_sd_over_seeds=float(cft.std()))
print("S1 5-fold cross-fit inside P (teacher):", res["S1_crossfit_in_P"], flush=True)


# ---------------------------------------------------------------- S3 / S4
def var_model(p, r, Mv):
    X = np.column_stack([np.ones(len(p)), 1.0 / Mv])
    return np.linalg.lstsq(X, (r / p) ** 2, rcond=None)[0]


VM = {m: var_model(pP[m], rP[m], MP) for m in MET}
res["var_model_P"] = {m: VM[m].tolist() for m in MET}


def tn_shift(p, s, U):
    """E[Y | Y < U] - p for Y ~ N(p, s^2), computed in log space"""
    b = (U - p) / s
    return -s * np.exp(norm.logpdf(b) - log_ndtr(b))


def upper(m, a, gmax):
    return np.full(len(a), gmax) if m == "gas" else gmax / a


def lowhi(m, a, b):
    if m == "gas":
        return b["euimin"] * a, np.full(len(a), b["gmax"])
    if m == "eui":
        return np.full(len(a), b["euimin"]), b["gmax"] / a
    return b["eamin"] * a, b["eamax"] * a


def run_chain(fit, steps, kfix=None, b=bP):
    """returns full-H prediction dict after the requested steps; S3's k fitted on rows `fit` of H (or fixed)"""
    cur = {m: base[m].copy() for m in MET}
    ks = {}
    if "S1" in steps:
        cur = {m: apply_s1(cur[m], MH, DP[m]) for m in MET}
    if "S3" in steps:
        for m in ("gas", "eui"):
            p = cur[m]
            s = p * np.sqrt(np.maximum(VM[m][0] + VM[m][1] / MH, 1e-8))
            x = tn_shift(p, s, upper(m, AH, b["gmax"]))
            k = kfix[m] if kfix is not None else float(x[fit] @ (y[m][fit] - p[fit]) / (x[fit] @ x[fit]))
            ks[m] = k
            cur[m] = p + k * x
    if "S4" in steps:
        for m in MET:
            lo, hi = lowhi(m, AH, b)
            cur[m] = np.clip(cur[m], lo, hi)
    return cur, ks


def crossfit(steps, seed, nfold=5):
    out = {m: np.empty(nH) for m in MET}
    ks = []
    for a, bb in KFold(nfold, shuffle=True, random_state=seed).split(allH):
        cur, k = run_chain(a, steps)
        ks.append(k)
        for m in MET:
            out[m][bb] = cur[m][bb]
    return out, ks


# S4 alone and S1+S4 need no H fit
for steps in (("S4",), ("S1", "S4")):
    cur, _ = run_chain(allH, steps)
    per = {m: nrmse(m, allH, cur[m]) - nrmse(m, allH, base[m]) for m in MET}
    res[f"{'+'.join(steps)}_H"] = dict(per=per, total=sum(per.values()) / 5,
                                       rows_changed={m: int((cur[m] != base[m]).sum()) for m in MET})
    print(f"{'+'.join(steps)} on H:", {m: round(v, 8) for m, v in per.items()}, "total", round(sum(per.values()) / 5, 8),
          res[f"{'+'.join(steps)}_H"]["rows_changed"], flush=True)

# k fitted on the P side only (P teacher, after S1 with P deltas applied to P) -> applied to H: fully out-of-sample
kP = {}
for m in ("gas", "eui"):
    p = apply_s1(pP[m], MP, DP[m])
    s = p * np.sqrt(np.maximum(VM[m][0] + VM[m][1] / MP, 1e-8))
    x = tn_shift(p, s, upper(m, AP, bP["gmax"]))
    kP[m] = float(x @ (yP[m] - p) / (x @ x))
res["k_fitted_on_P"] = kP
for lab, kf in (("kP", kP), ("k1", {"gas": 1.0, "eui": 1.0})):
    for steps in (("S3",), ("S1", "S3", "S4")):
        cur, _ = run_chain(allH, steps, kfix=kf)
        per = {m: nrmse(m, allH, cur[m]) - nrmse(m, allH, base[m]) for m in MET}
        res[f"{'+'.join(steps)}_{lab}_H"] = dict(per=per, total=sum(per.values()) / 5)
        print(f"{'+'.join(steps)} with {lab} {kf}:", {m: round(v, 8) for m, v in per.items()},
              "total", round(sum(per.values()) / 5, 8), flush=True)

# cross-fitting with seeds 1..60 (auditor used seed 0 only), steps S3 alone and the full S1+S3+S4
for steps in (("S3",), ("S1", "S3", "S4")):
    tots, pers, kk = [], [], []
    for seed in range(0, 61):
        out, ks = crossfit(steps, seed)
        per = {m: nrmse(m, allH, out[m]) - nrmse(m, allH, base[m]) for m in MET}
        if seed == 0:
            res[f"{'+'.join(steps)}_crossfit_seed0"] = dict(per=per, total=sum(per.values()) / 5)
            continue
        pers.append(per)
        tots.append(sum(per.values()) / 5)
        kk.extend(ks)
    tots = np.array(tots)
    res[f"{'+'.join(steps)}_crossfit_seeds1_60"] = dict(
        total_mean=float(tots.mean()), total_sd=float(tots.std()), total_min=float(tots.min()), total_max=float(tots.max()),
        per_mean={m: float(np.mean([p[m] for p in pers])) for m in MET},
        k_mean={m: float(np.mean([k[m] for k in kk])) for m in ("gas", "eui")},
        k_sd={m: float(np.std([k[m] for k in kk])) for m in ("gas", "eui")})
    print(f"crossfit {'+'.join(steps)} seed0", res[f"{'+'.join(steps)}_crossfit_seed0"], flush=True)
    print(f"crossfit {'+'.join(steps)} seeds1-60", res[f"{'+'.join(steps)}_crossfit_seeds1_60"], flush=True)

# split-halves with new seeds (fit S3's k on one half, score the change on the other half), both directions
for steps in (("S3",), ("S1", "S3", "S4"), ("S1",)):
    tots = []
    for rep in range(100):
        perm = np.random.default_rng(9000 + rep).permutation(nH)
        h1, h2 = np.sort(perm[: nH // 2]), np.sort(perm[nH // 2:])
        for a, b in ((h1, h2), (h2, h1)):
            cur, _ = run_chain(a, steps)
            tots.append(sum(nrmse(m, b, cur[m]) - nrmse(m, b, base[m]) for m in MET) / 5)
    tots = np.array(tots)
    res[f"{'+'.join(steps)}_splithalf"] = dict(mean=float(tots.mean()), sd=float(tots.std()), frac_neg=float((tots < 0).mean()))
    print(f"split-half {'+'.join(steps)}:", res[f"{'+'.join(steps)}_splithalf"], flush=True)

# public-size bootstrap of the full cross-fitted chain (seed 0 predictions, the auditor's estimate)
out0, _ = crossfit(("S1", "S3", "S4"), 0)
e_new = {m: (y[m] - out0[m]) ** 2 for m in MET}
bp = boot(e_new, e_old, 55196, 2000, 13)
bt = boot(e_new, e_old, nH, 2000, 14)
res["chain_boot_public_size"] = dict(mean=float(bp.mean()), sd=float(bp.std()), frac_neg=float((bp < 0).mean()),
                                     q05=float(np.quantile(bp, .05)), q95=float(np.quantile(bp, .95)))
res["chain_boot_H"] = dict(mean=float(bt.mean()), sd=float(bt.std()), frac_neg=float((bt < 0).mean()),
                           q05=float(np.quantile(bt, .05)), q95=float(np.quantile(bt, .95)))
print("chain bootstrap |H|:", res["chain_boot_H"], "| public-size:", res["chain_boot_public_size"], flush=True)
json.dump(res, open(f"{OUT}/r1_chainH.json", "w"), indent=1)
print("wrote", f"{OUT}/r1_chainH.json")
