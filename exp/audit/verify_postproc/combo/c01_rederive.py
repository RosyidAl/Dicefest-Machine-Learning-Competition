"""verify_postproc/combo c01: independent re-derivation of the combined post-processing chain S1 + S3 + S4 on the
honest holdout chain (exp/audit/chain_H.npz), written from scratch (no code shared with exp/audit/postproc/).

S0  base   = clip(corrected, 0)
S1  p *= 1 + d_g for H rows with M in {2,3} (g1) and M == 4 (g2); d_g = sum(pP rP) / sum(pP^2) over P rows of the group
    (P-side OOF teacher residuals; no H label involved)
S3  gas, eui: p += k * x, x = -s phi(b) / Phi(b), b = (hi - p) / s, s = p sqrt(a0 + b0 / M) ((a0, b0): lstsq of
    (rP / pP)^2 on [1, 1/M] over P); hi = max gas over P (gas) or that / A (eui); k = lstsq on H training folds only
S4  hard clip to P-derived bounds: gas in [min eui * A, max gas], eui in [min eui, max gas / A], elec in
    [min(elec/A) A, max(elec/A) A]

Reports: per-metric NRMSE change and TOTAL (= sum of the three per-metric changes / 5), per step; robustness over
fold seeds; nonparametric bootstrap over H rows (scoring noise); simulated public-LB subsets (55,196 rows drawn from H);
how concentrated the gain is (top rows); per-group contributions of S1.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/combo/c01_rederive.py
"""
import json
import sys

import numpy as np
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
assert abs(SIG["gas"] - 151750.04) < 1e-6 and abs(SIG["eui"] - 40.65) < 1e-9 and abs(SIG["elec"] - 41725.37) < 1e-6
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/combo"

zf = np.load("exp/audit/chain_H.npz", allow_pickle=False)
z = {k: zf[k] for k in zf.files if not k.endswith("_F")}
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
assert len(np.intersect1d(P, H)) == 0
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
nH = len(H)
y = {m: z[f"{m}_y"] for m in MET}
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in MET}
# sanity: gas == eui * A on both sides (identity in the data)
assert np.allclose(y["gas"], y["eui"] * AH, rtol=1e-9)
assert np.allclose(yP["gas"], yP["eui"] * AP, rtol=1e-6)
GMAX = yP["gas"].max()
EUIMIN = yP["eui"].min()
EAMIN, EAMAX = (yP["elec"] / AP).min(), (yP["elec"] / AP).max()
print(f"P bounds: gas_max {GMAX:.2f} eui_min {EUIMIN:.4f} elecA [{EAMIN:.4f}, {EAMAX:.4f}]; H y max gas {y['gas'].max():.2f}")
LO = {"gas": EUIMIN * AH, "eui": np.full(nH, EUIMIN), "elec": EAMIN * AH}
HI = {"gas": np.full(nH, GMAX), "eui": GMAX / AH, "elec": EAMAX * AH}
GROUPS = ((2, 3), (4, 4))
DELTA = {}
for m in MET:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    DELTA[m] = []
    for lo, hi in GROUPS:
        g = (MP >= lo) & (MP <= hi)
        DELTA[m].append(float(np.dot(pP[g], rP[g]) / np.dot(pP[g], pP[g])))
VAR = {}
for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    X = np.column_stack([np.ones(len(pP)), 1.0 / MP])
    VAR[m] = np.linalg.lstsq(X, (rP / pP) ** 2, rcond=None)[0]
print("S1 deltas (P-OOF)", {m: np.round(v, 5).tolist() for m, v in DELTA.items()})
print("S3 variance model (P)", {m: np.round(v, 5).tolist() for m, v in VAR.items()})
print("H group sizes", {f"{lo}-{hi}": int(((MH >= lo) & (MH <= hi)).sum()) for lo, hi in GROUPS},
      "P group sizes", {f"{lo}-{hi}": int(((MP >= lo) & (MP <= hi)).sum()) for lo, hi in GROUPS})


def s1(m, p):
    q = p.copy()
    for (lo, hi), d in zip(GROUPS, DELTA[m]):
        g = (MH >= lo) & (MH <= hi)
        q[g] = q[g] * (1.0 + d)
    return q


def s3_term(m, p):
    a0, b0 = VAR[m]
    s = p * np.sqrt(np.maximum(a0 + b0 / MH, 1e-8))
    b = (HI[m] - p) / s
    # phi(b) / Phi(b) computed in log space (stable for very negative b)
    ratio = np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))
    return -s * ratio


def nrmse(m, p, idx=None):
    if idx is None:
        return float(np.sqrt(np.mean((y[m] - p) ** 2)) / SIG[m])
    return float(np.sqrt(np.mean((y[m][idx] - p[idx]) ** 2)) / SIG[m])


def run_chain(seed, steps=("S1", "S3", "S4")):
    """returns dict step -> metric -> full-H cross-fitted predictions, and the per-fold k"""
    base = {m: np.maximum(z[f"{m}_corrected"], 0.0) for m in MET}
    out = {"S0": base}
    cur = {m: (s1(m, base[m]) if "S1" in steps else base[m].copy()) for m in MET}
    out["S1"] = {m: cur[m].copy() for m in MET}
    ks = {}
    if "S3" in steps:
        for m in ("gas", "eui"):
            x = s3_term(m, cur[m])
            r = y[m] - cur[m]
            q = cur[m].copy()
            ks[m] = []
            for a, b in KFold(5, shuffle=True, random_state=seed).split(x):
                k = float(np.dot(x[a], r[a]) / np.dot(x[a], x[a]))
                ks[m].append(round(k, 4))
                q[b] = cur[m][b] + k * x[b]
            cur[m] = q
    out["S3"] = {m: cur[m].copy() for m in MET}
    if "S4" in steps:
        cur = {m: np.clip(cur[m], LO[m], HI[m]) for m in MET}
    out["S4"] = {m: cur[m].copy() for m in MET}
    return out, ks


res, ks = run_chain(0)
s0 = {m: nrmse(m, res["S0"][m]) for m in MET}
print("S0 (base) NRMSE", {m: round(v, 6) for m, v in s0.items()})
prev = s0
summary = {"base": s0}
for st in ("S1", "S3", "S4"):
    sc = {m: nrmse(m, res[st][m]) for m in MET}
    step = {m: sc[m] - prev[m] for m in MET}
    cum = {m: sc[m] - s0[m] for m in MET}
    print(f"{st}: step " + " ".join(f"{m} {step[m]:+.7f}" for m in MET) + f" | step total {sum(step.values()) / 5:+.8f}"
          f" | cum total {sum(cum.values()) / 5:+.8f}")
    prev = sc
    summary[st] = {"step": step, "cum_total": sum(cum.values()) / 5}
print("k per fold (seed 0)", ks)
final = res["S4"]
dm = {m: nrmse(m, final[m]) - s0[m] for m in MET}
TOT = sum(dm.values()) / 5
print(f"COMBINED 5-fold (seed 0): " + " ".join(f"{m} {dm[m]:+.7f}" for m in MET) + f" TOTAL {TOT:+.8f}")
summary["combined_seed0"] = {"per_metric": dm, "total": TOT}

# robustness over fold seeds for the S3 k cross-fit
tots = []
for seed in range(1, 11):
    r_, _ = run_chain(seed)
    tots.append(sum(nrmse(m, r_["S4"][m]) - s0[m] for m in MET) / 5)
print(f"TOTAL over fold seeds 1..10: mean {np.mean(tots):+.8f} min {np.min(tots):+.8f} max {np.max(tots):+.8f}")
summary["total_seeds"] = tots

# bootstrap over H rows: scoring noise of the full-H estimate
e0 = {m: (y[m] - res["S0"][m]) ** 2 for m in MET}
e1 = {m: (y[m] - final[m]) ** 2 for m in MET}
rng = np.random.default_rng(12345)
B = 2000
bt = np.empty(B)
for i in range(B):
    idx = rng.integers(0, nH, nH)
    bt[i] = sum(np.sqrt(e1[m][idx].mean()) / SIG[m] - np.sqrt(e0[m][idx].mean()) / SIG[m] for m in MET) / 5
print(f"bootstrap (H rows, {B}x): total mean {bt.mean():+.8f} sd {bt.std():.8f} 2.5% {np.quantile(bt, 0.025):+.8f} "
      f"97.5% {np.quantile(bt, 0.975):+.8f} frac>=0 {np.mean(bt >= 0):.4f}")
summary["bootstrap"] = {"mean": bt.mean(), "sd": bt.std(), "q025": np.quantile(bt, 0.025), "q975": np.quantile(bt, 0.975),
                        "frac_ge0": float(np.mean(bt >= 0))}

# simulated public-LB subsets: 30% of test rows (55,196), drawn without replacement from H
npub = int(round(0.3 * 183987))
sim = np.empty(B)
for i in range(B):
    idx = rng.choice(nH, npub, replace=False)
    sim[i] = sum(np.sqrt(e1[m][idx].mean()) / SIG[m] - np.sqrt(e0[m][idx].mean()) / SIG[m] for m in MET) / 5
print(f"public-size subsets (n={npub}): mean {sim.mean():+.8f} sd {sim.std():.8f} frac>=0 {np.mean(sim >= 0):.3f} "
      f"frac<=-0.00011 (enough for 0.24912) {np.mean(sim <= -0.00011):.3f}")
summary["public_sim"] = {"mean": sim.mean(), "sd": sim.std(), "frac_ge0": float(np.mean(sim >= 0))}

# concentration: per-row contribution to the MSE change, per metric (in total-score units, linearised)
print("concentration of the gain (linearised per-row contribution to the TOTAL):")
contrib = np.zeros(nH)
for m in MET:
    rm = np.sqrt(e0[m].mean())
    contrib += (e1[m] - e0[m]) / (2 * rm * nH) / SIG[m] / 5
o = np.argsort(contrib)
print(f"   sum {contrib.sum():+.8f}; rows changed {int((contrib != 0).sum())}; best 10 rows {contrib[o[:10]].sum():+.8f}; "
      f"worst 10 rows {contrib[o[-10:]].sum():+.8f}; without best 10 {contrib.sum() - contrib[o[:10]].sum():+.8f}; "
      f"without best 25 {contrib.sum() - contrib[o[:25]].sum():+.8f}")
for name, g in (("M 2-3", (MH >= 2) & (MH <= 3)), ("M 4", MH == 4), ("M>=5", MH >= 5)):
    print(f"   {name:6s}: n {int(g.sum()):6d} contribution {contrib[g].sum():+.8f}")
summary["concentration"] = {"best10": float(contrib[o[:10]].sum()), "worst10": float(contrib[o[-10:]].sum())}
json.dump(summary, open(f"{OUT}/c01_rederive.json", "w"), indent=1, default=float)
