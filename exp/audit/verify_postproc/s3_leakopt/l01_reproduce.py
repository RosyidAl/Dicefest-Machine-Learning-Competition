"""Adversarial re-derivation (leakage / optimism lens) of the S3 soft cap claim on the honest holdout chain H.

Independent code (does not import the postproc auditor's scripts). Inputs: exp/audit/chain_H.npz only, plus the raw
train gas column (pandas usecols) for an integrity check when MemAvailable allows it.

Checks
  0. integrity: P/H partition, y on P = pP + rP equals raw train targets, where the train gas maximum sits (P or H)
  1. S1 (small-M multiplicative shift, deltas from P-OOF) then S3 on gas and eui
  2. k variants: in-sample on all H (reproduces the claimed constants), 5-fold cross-fitted on H (50 seeds),
     split-half (100 repeats), k fitted on P-OOF teacher only (no H label touched), k = 1 (pure theory)
  3. bound variants: T from P labels only, T = global train max (includes one H label), T = 1e6
  4. per-row decomposition of the gain (concentration, leave-top-rows-out)
  5. bootstrap of the H gain for the H-free rule (k, s-model, T all from P)
  6. calibration by standardized distance to the cap before / after S3 (over-correction check)
  7. same rule on the H teacher (no residual correction) for robustness

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/s3_leakopt/l01_reproduce.py
"""
import json
import resource
import sys
import time

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

t0 = time.time()
OUT = "exp/audit/verify_postproc/s3_leakopt"
GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
MET = ("gas", "eui")
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
nH = len(H)
res = {}

# ---------------- 0. integrity
assert len(np.intersect1d(P, H)) == 0 and len(P) + len(H) == len(A)
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in MET}
yH = {m: z[f"{m}_y"] for m in MET}
print("gas = eui * A on H: max rel diff", float(np.max(np.abs(yH["gas"] - yH["eui"] * AH) / yH["gas"])))
print("gas = eui * A on P: max rel diff", float(np.max(np.abs(yP["gas"] - yP["eui"] * AP) / yP["gas"])))
GMAX_P, GMAX_H = float(yP["gas"].max()), float(yH["gas"].max())
print(f"gas max on P {GMAX_P:.4f} | on H {GMAX_H:.4f} | global {max(GMAX_P, GMAX_H):.4f}")
print("H rows with gas > max(P):", int((yH["gas"] > GMAX_P).sum()))
mem_kb = int([ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0])
if mem_kb >= 9_000_000:
    import pandas as pd
    raw = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI])
    for m, col in (("gas", GAS), ("eui", EUI)):
        v = raw[col].values
        print(f"raw {m}: max rel diff P {np.max(np.abs(v[P] - yP[m]) / v[P]):.2e} H {np.max(np.abs(v[H] - yH[m]) / v[H]):.2e}")
    print(f"raw train gas max {raw[GAS].max():.4f} (index in H: {int(np.argmax(raw[GAS].values)) in set(H.tolist())})")
    del raw
else:
    print(f"MemAvailable {mem_kb} kB < 9e6: raw-target integrity check skipped")


def nr(m, y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def ls_delta(p, r):
    return float(p @ r / (p @ p))


def mills_shift(p, s, T):
    """E[Y | Y < T] - mu for Y ~ N(mu, s^2) evaluated at mu = p: -s phi(b)/Phi(b), b = (T - p)/s (log-space, stable)"""
    b = (T - p) / s
    return -s * np.exp(norm.logpdf(b) - norm.logcdf(b))


# s-model from P-OOF relative residuals: (r/p)^2 ~ a0 + b0 / M
SD = {}
for m in MET:
    pp, rr = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pp)), 1.0 / MP]), (rr / pp) ** 2, rcond=None)[0]
print("s-model (a0, b0) from P:", {m: np.round(v, 5).tolist() for m, v in SD.items()})


def sdev(m, p, Mv):
    return p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / Mv, 1e-8, None))


# ---------------- 1. base = clip(corrected, 0) then S1 with P-OOF deltas (M 2-3, M 4)
base0, base = {}, {}
for m in MET:
    p = np.clip(z[f"{m}_corrected"], 0, None)
    base0[m] = p.copy()
    pp, rr = z[f"{m}_pP"], z[f"{m}_rP"]
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + ls_delta(pp[gP], rr[gP])
    base[m] = p
S0 = {m: nr(m, yH[m], base0[m]) for m in MET}
S1 = {m: nr(m, yH[m], base[m]) for m in MET}
print("S0", {m: round(v, 6) for m, v in S0.items()}, "| after S1", {m: round(v, 6) for m, v in S1.items()})

TV = {"T_P": GMAX_P, "T_global": max(GMAX_P, GMAX_H), "T_1e6": 1e6}


def xH(m, T):
    p = base[m]
    return mills_shift(p, sdev(m, p, MH), T if m == "gas" else T / AH)


def apply(m, p, x, k, T):
    """S3 then S4-style hard clip at the bound (as in the claimed chain)"""
    hi = T if m == "gas" else T / AH
    return np.minimum(p + k * x, hi)


# ---------------- 2./3. k variants x bound variants
for tname, T in TV.items():
    out = {}
    for m in MET:
        p, y = base[m], yH[m]
        x = xH(m, T)
        r = y - p
        k_ins = float(x @ r / (x @ x))
        # k from P-OOF teacher only (no H label): x on P rows with T from P labels
        pp = z[f"{m}_pP"]
        xP = mills_shift(pp, sdev(m, pp, MP), T if m == "gas" else T / AP)
        k_P = float(xP @ z[f"{m}_rP"] / (xP @ xP))
        d = {}
        d["k_ins_H"] = (k_ins, nr(m, y, apply(m, p, x, k_ins, T)) - S1[m])
        d["k_P"] = (k_P, nr(m, y, apply(m, p, x, k_P, T)) - S1[m])
        d["k_1"] = (1.0, nr(m, y, apply(m, p, x, 1.0, T)) - S1[m])
        d["k_0"] = (0.0, nr(m, y, apply(m, p, x, 0.0, T)) - S1[m])  # hard clip only
        # 5-fold cross-fitted on H, 50 seeds
        cf = []
        for seed in range(50):
            pc = np.empty(nH)
            for a, b in KFold(5, shuffle=True, random_state=seed).split(p):
                k = float(x[a] @ r[a] / (x[a] @ x[a]))
                pc[b] = apply(m, p[b], x[b], k, T) if m == "gas" else np.minimum(p[b] + k * x[b], T / AH[b])
            cf.append(nr(m, y, pc) - S1[m])
        d["cf5_seed0"] = (None, cf[0])
        d["cf5_mean50"] = (None, float(np.mean(cf)))
        d["cf5_sd50"] = (None, float(np.std(cf)))
        out[m] = d
    tot = {key: (out["gas"][key][1] + out["eui"][key][1]) / 5 for key in out["gas"]}
    print(f"== bound {tname} = {T:.4f}")
    for m in MET:
        print("   " + m + ": " + " | ".join(f"{key} k={v[0]:.4f} d={v[1]:+.7f}" if v[0] is not None else f"{key} d={v[1]:+.7f}"
                                          for key, v in out[m].items()))
    print("   TOTAL: " + " | ".join(f"{key} {v:+.8f}" for key, v in tot.items() if key != "cf5_sd50"))
    res[tname] = {"per_metric": {m: {k2: list(v) for k2, v in out[m].items()} for m in MET}, "total": tot}

# split-half (fit k on one half, score the other half; both directions), T from P
T = GMAX_P
sh = []
for rep in range(100):
    perm = np.random.default_rng(1000 + rep).permutation(nH)
    h1, h2 = perm[: nH // 2], perm[nH // 2:]
    for a, b in ((h1, h2), (h2, h1)):
        tot = 0.0
        for m in MET:
            p, y = base[m], yH[m]
            x = xH(m, T)
            r = y - p
            k = float(x[a] @ r[a] / (x[a] @ x[a]))
            hi = T if m == "gas" else T / AH[b]
            tot += (nr(m, y[b], np.minimum(p[b] + k * x[b], hi)) - nr(m, y[b], p[b])) / 5
        sh.append(tot)
sh = np.array(sh)
print(f"split-half x200 (T from P): total mean {sh.mean():+.8f} sd {sh.std():.8f} frac<0 {np.mean(sh < 0):.3f}")
res["split_half"] = {"mean": float(sh.mean()), "sd": float(sh.std()), "frac_neg": float(np.mean(sh < 0))}

# ---------------- 4./5. H-free rule (k from P-OOF teacher, T and s from P): per-row decomposition + bootstrap
kP = {m: res["T_P"]["per_metric"][m]["k_P"][0] for m in MET}
newP = {m: apply(m, base[m], xH(m, GMAX_P), kP[m], GMAX_P) for m in MET}
contrib = np.zeros(nH)
for m in MET:
    e0, e1 = yH[m] - base[m], yH[m] - newP[m]
    rmse0 = np.sqrt(np.mean(e0 ** 2))
    c = (e1 ** 2 - e0 ** 2) / (2 * nH * rmse0 * SIG[m]) / 5  # first-order per-row change in total-score units
    contrib += c
    ch = np.abs(newP[m] - base[m]) / SIG[m]
    print(f"{m}: k_P {kP[m]:.4f}; rows changed >0.001 sigma {int((ch > 1e-3).sum())}, >0.01 sigma {int((ch > 1e-2).sum())}, "
          f">0.1 sigma {int((ch > 0.1).sum())}; first-order sum {c.sum():+.8f}; helped {int((c < -1e-12).sum())} hurt {int((c > 1e-12).sum())}")
order = np.argsort(contrib)
print(f"total first-order {contrib.sum():+.8f}; most-helped 10 rows {np.round(contrib[order[:10]] * 1e6, 3).tolist()} (x1e-6)")
print(f"most-hurt 10 rows {np.round(contrib[order[::-1][:10]] * 1e6, 3).tolist()} (x1e-6)")
for n in (1, 3, 5, 10, 20, 40):
    print(f"   without the {n:2d} most-helped rows: {contrib.sum() - contrib[order[:n]].sum():+.8f}"
          f" | without the {n:2d} most-helped AND most-hurt rows: {contrib.sum() - contrib[order[:n]].sum() - contrib[order[::-1][:n]].sum():+.8f}")
res["Hfree_rule"] = {"kP": kP, "first_order_total": float(contrib.sum())}

rng = np.random.default_rng(12345)
bt = []
e0 = {m: yH[m] - base[m] for m in MET}
e1 = {m: yH[m] - newP[m] for m in MET}
for _ in range(2000):
    i = rng.integers(0, nH, nH)
    bt.append(sum((np.sqrt(np.mean(e1[m][i] ** 2)) - np.sqrt(np.mean(e0[m][i] ** 2))) / SIG[m] for m in MET) / 5)
bt = np.array(bt)
print(f"bootstrap H-free rule: mean {bt.mean():+.8f} sd {bt.std():.8f} 2.5/97.5% [{np.quantile(bt, .025):+.8f}, "
      f"{np.quantile(bt, .975):+.8f}] P(<0) {np.mean(bt < 0):.3f}")
res["Hfree_bootstrap"] = {"mean": float(bt.mean()), "sd": float(bt.std()), "q025": float(np.quantile(bt, .025)),
                          "q975": float(np.quantile(bt, .975)), "frac_neg": float(np.mean(bt < 0))}

# ---------------- 6. calibration by standardized distance before / after (claimed constants, T from P)
KCL = {"gas": 0.5046, "eui": 0.6574}
for m in MET:
    p, y = base[m], yH[m]
    s = sdev(m, p, MH)
    Tm = GMAX_P if m == "gas" else GMAX_P / AH
    dist = (Tm - p) / s
    pn = apply(m, p, xH(m, GMAX_P), KCL[m], GMAX_P)
    print(f"== calibration {m} (k {KCL[m]}): bin | n | mean resid before (t) | after (t)")
    for lo, hi in ((-9, 0), (0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)):
        sel = (dist >= lo) & (dist < hi)
        r0, r1 = y[sel] - p[sel], y[sel] - pn[sel]
        se0, se1 = r0.std() / np.sqrt(sel.sum()), r1.std() / np.sqrt(sel.sum())
        print(f"   [{lo},{hi}) n {sel.sum():6d} before {r0.mean():+12.3f} ({r0.mean() / se0:+5.1f}) after {r1.mean():+12.3f} ({r1.mean() / se1:+5.1f})")

# ---------------- 7. same rule on the teacher (no residual correction): robustness
for m in MET:
    p = np.clip(z[f"{m}_teacher"], 0, None)
    pp, rr = z[f"{m}_pP"], z[f"{m}_rP"]
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + ls_delta(pp[gP], rr[gP])
    x = mills_shift(p, sdev(m, p, MH), GMAX_P if m == "gas" else GMAX_P / AH)
    r = yH[m] - p
    pc = np.empty(nH)
    for a, b in KFold(5, shuffle=True, random_state=0).split(p):
        k = float(x[a] @ r[a] / (x[a] @ x[a]))
        pc[b] = np.minimum(p[b] + k * x[b], GMAX_P if m == "gas" else GMAX_P / AH[b])
    print(f"teacher-only {m}: k in-sample {float(x @ r / (x @ x)):.4f}; cross-fitted d {nr(m, yH[m], pc) - nr(m, yH[m], p):+.7f}"
          f"; with k_P {nr(m, yH[m], np.minimum(p + kP[m] * x, GMAX_P if m == 'gas' else GMAX_P / AH)) - nr(m, yH[m], p):+.7f}")

json.dump(res, open(f"{OUT}/l01_reproduce.json", "w"), indent=1, default=float)
print(f"wrote {OUT}/l01_reproduce.json; wall {time.time() - t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
