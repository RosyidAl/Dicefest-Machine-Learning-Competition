"""verify2_corrhyper v01: independent reproduction of corrhyper E2 (group features at 2-3 shrinkage levels).
Own implementation (pandas groupby + sorted searchsorted window), different honest splits:
  A) fixed candidates, lambda cross-fitted on H with 10-fold KFold, seeds 101..105 (no selection).
  B) select-on-one-half / score-on-other, half seeds 21..30, inner 5-fold KFold seed 7.
Total change = per-target NRMSE change / 5."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

OUTD = "exp/audit/verify2_corrhyper/"
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
tr = pd.read_parquet("cache/train.parquet", columns=["zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"])
zone = tr["zone_region_id"].astype(str).values
cell = (tr["env_climate_01"].astype(str) + "|" + tr["env_climate_03"].astype(str) + "|" + tr["env_climate_05"].astype(str)).values
del tr


def shr_group(keys, num, den, K):
    df = pd.DataFrame({"k": keys[P], "s": num, "d": den, "n": 1.0}).groupby("k").sum()
    s = pd.Series(keys[H]).map(df["s"]).fillna(0).values
    d = pd.Series(keys[H]).map(df["d"]).fillna(0).values
    n = pd.Series(keys[H]).map(df["n"]).fillna(0).values
    mean = np.divide(s, d, out=np.zeros_like(s), where=d > 0)
    return n / (n + K) * mean


def shr_window(num, den, K, W=30):
    o = np.argsort(subn[P], kind="stable")
    sp = subn[P][o]
    cs = [np.r_[0, np.cumsum(v[o])] for v in (num, den, np.ones(len(o)))]
    lo = np.searchsorted(sp, subn[H] - W, "left")
    hi = np.searchsorted(sp, subn[H] + W, "right")
    s, d, n = (c[hi] - c[lo] for c in cs)
    mean = np.divide(s, d, out=np.zeros_like(s), where=d > 0)
    return n / (n + K) * mean


def cf_nrmse(F, r, sig, seed, nf):
    pc = np.zeros(len(r))
    for a, b in KFold(nf, shuffle=True, random_state=seed).split(F):
        pc[b] = F[b] @ np.linalg.lstsq(F[a], r[a], rcond=None)[0]
    return np.sqrt(np.mean((r - pc) ** 2)) / sig


log = open(OUTD + "v01_twolevel.log", "w")
def L(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()

res = {}
halves = [np.random.default_rng(s).permutation(len(H)) % 2 for s in range(21, 31)]
for m in ("gas", "eui", "elec"):
    F = z[f"{m}_F"].astype(np.float64)
    rP, pP, pH, y = z[f"{m}_rP"], z[f"{m}_pP"], z[f"{m}_teacher"], z[f"{m}_y"]
    r = y - pH
    sig = SIG[m]
    if m == "gas":
        num, den, mult = rP, A[P], A[H]
    elif m == "eui":
        num, den, mult = A[P] * rP, A[P], np.ones(len(H))
    else:
        num, den, mult = rP, pP, pH
    G = lambda K: np.column_stack([shr_group(zone, num, den, K) * mult, shr_group(cell, num, den, K) * mult,
                                   shr_window(num, den, K) * mult])
    Gs = {K: G(K) for K in (3, 10, 30, 100, 300)}
    err = np.abs(Gs[30] - F[:, :3]).max() / np.abs(F[:, :3]).max()
    corr_chk = np.sqrt(np.mean((y - z[f"{m}_corrected"]) ** 2)) / sig
    L(f"== {m}: my group cols vs stored F max rel err {err:.2e}; stored corrected NRMSE {corr_chk:.6f}")
    Kn = F[:, 3:]
    cands = {"base": F}
    for lo_, hi_ in ((3, 100), (3, 300), (10, 100), (10, 300), (10, 30), (30, 100)):
        cands[f"{lo_}/{hi_}"] = np.column_stack([Gs[lo_], Gs[hi_], Kn])
    cands["3/30/300"] = np.column_stack([Gs[3], Gs[30], Gs[300], Kn])
    # A) fixed candidates, 10-fold, 5 seeds
    A_res = {}
    for k, Fc in cands.items():
        A_res[k] = [cf_nrmse(Fc, r, sig, s, 10) for s in range(101, 106)]
    b = np.array(A_res["base"])
    dA = {k: float(np.mean(np.array(v) - b)) for k, v in A_res.items()}
    sdA = {k: float(np.std(np.array(v) - b)) for k, v in A_res.items()}
    L(f"  [A fixed, 10-fold x5 seeds] base {b.mean():.6f}; delta NRMSE: " +
      ", ".join(f"{k} {dA[k]:+.2e}(sd {sdA[k]:.1e})" for k in cands if k != "base"))
    # B) select on one half, score on the other
    hon, picks, fixed_other = [], [], {k: [] for k in cands}
    for hv in halves:
        sc = {k: [cf_nrmse(Fc[hv == h], r[hv == h], sig, 7, 5) for h in (0, 1)] for k, Fc in cands.items()}
        for h in (0, 1):
            g = min(sc, key=lambda k: sc[k][h])
            picks.append(g)
            hon.append(sc[g][1 - h] - sc["base"][1 - h])
            for k in cands:
                fixed_other[k].append(sc[k][1 - h] - sc["base"][1 - h])
    L(f"  [B select-half, seeds 21-30] honest mean {np.mean(hon):+.2e}; frac<0 {np.mean(np.array(hon) < 0):.2f}; picks {pd.Series(picks).value_counts().to_dict()}")
    L(f"  [B per-half fixed deltas] " + ", ".join(f"{k} {np.mean(v):+.2e}" for k, v in fixed_other.items() if k != "base"))
    res[m] = dict(err=float(err), A=dA, A_sd=sdA, B_honest=float(np.mean(hon)), B_list=[float(x) for x in hon],
                  picks=picks, B_fixed={k: float(np.mean(v)) for k, v in fixed_other.items()})

totB = sum(res[m]["B_honest"] for m in res) / 5
L(f"TOTAL B (selection honest): {totB:+.8f}")
for k in res["gas"]["A"]:
    if k == "base":
        continue
    L(f"TOTAL A fixed {k}: {sum(res[m]['A'][k] for m in res) / 5:+.8f}  (per target gas {res['gas']['A'][k]:+.1e} eui {res['eui']['A'][k]:+.1e} elec {res['elec']['A'][k]:+.1e})")
best_per_target = sum(min(res[m]["A"].values()) for m in res) / 5
L(f"TOTAL A oracle (best fixed per target, optimistic): {best_per_target:+.8f}")
res["totB"] = totB
json.dump(res, open(OUTD + "v01_twolevel.json", "w"), indent=1)
