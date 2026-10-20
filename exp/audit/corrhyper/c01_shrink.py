"""corrhyper c01: residual-correction hyperparameters re-tuned honestly on the holdout chain (exp/audit/chain_H.npz).

New angles (exp/audit/tried/group_k.py already covered a per-group K x WIN grid and extra windows):
  E0  reproduction of the stored chain (F float32) and of the 3 group columns from rP.
  E1  kNN shrinkage KSHR (n/(n+KSHR)), re-tuned exactly by rescaling the stored kNN columns:
      col(K') = col(5) * [c/(c+K')] / [c/(c+5)], c = min(k, #P rows with |sub_n diff| <= W). Verified in c00_verify.
  E2  group features at two shrinkage levels at once (6 group columns) and at three.
  E3  Kish effective-n shrinkage n_eff=(sum den)^2/sum den^2 instead of the row count.
  E4  joint E1 x (group K, WIN) grid.
Honest scoring: lambda 5-fold cross-fitted lstsq on H (as the chain). Choices checked by split-half (choose on one
half of H, score on the other, both cross-fitted inside the half), over 4 random half splits (seed 11 = previous).
All effects reported as change of the TOTAL score (metric NRMSE change / 5).
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
CFG = json.load(open("exp/v19vce/resid_knn.json"))["configs"]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
tr = pd.read_parquet("cache/train.parquet", columns=["zone_region_id", "env_climate_01", "env_climate_03",
                                                     "env_climate_05"])
zone = pd.factorize(tr["zone_region_id"])[0]
cell = pd.factorize(pd.Series(list(zip(tr["env_climate_01"], tr["env_climate_03"], tr["env_climate_05"]))))[0]
del tr
HALF_SEEDS = (11, 12, 13, 14)
HALVES = [np.random.default_rng(s).permutation(len(H)) % 2 for s in HALF_SEEDS]
OUT = {}


def group_mean(gf, ga, num, den, K, w=None):
    ng = max(gf.max(), ga.max()) + 1
    s, dn, n = (np.bincount(gf, v, ng)[ga] for v in (num, den, np.ones(len(gf))))
    if w == "kish":
        d2 = np.bincount(gf, den ** 2, ng)[ga]
        n = np.where(d2 > 0, dn ** 2 / np.where(d2 > 0, d2, 1), 0.0)
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def window_mean(sf, sa, num, den, K, WIN, w=None):
    m = max(sf.max(), sa.max()) + 1
    lo, hi = np.clip(sa - WIN, 0, m), np.clip(sa + WIN + 1, 0, m)
    cs = lambda v: (c := np.r_[0, np.cumsum(np.bincount(sf, v, m))])[hi] - c[lo]  # noqa: E731
    s, dn, n = cs(num), cs(den), cs(np.ones(len(sf)))
    if w == "kish":
        d2 = cs(den ** 2)
        n = np.where(d2 > 0, dn ** 2 / np.where(d2 > 0, d2, 1), 0.0)
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def pool_count(W, k):
    """number of valid neighbours used by resid_knn.knn_mean for each H row: min(k, #P rows within +-W sub_n)"""
    so = np.sort(subn[P])
    sa = subn[H]
    pool = np.searchsorted(so, sa + W, "right") - np.searchsorted(so, sa - W, "left")
    return np.minimum(k, pool).astype(float)


def cv(F, r, sig, idx=None):
    if idx is not None:
        F, r = F[idx], r[idx]
    pc = np.zeros(len(r))
    for a, b in KFold(5, shuffle=True, random_state=0).split(F):
        pc[b] = F[b] @ np.linalg.lstsq(F[a], r[a], rcond=None)[0]
    return np.sqrt(np.mean((r - pc) ** 2)) / sig


def evaluate(name, cands, base_key, r, sig, log):
    """cands: {key: F}; returns all-H change of the best key and split-half honest change (mean over 4 splits)"""
    full = {k: cv(F, r, sig) for k, F in cands.items()}
    half = {k: [[cv(F, r, sig, hv == h) for h in (0, 1)] for hv in HALVES] for k, F in cands.items()}
    b0 = full[base_key]
    best = min(full, key=full.get)
    hon, picks = [], []
    for si in range(len(HALVES)):
        for h in (0, 1):
            g = min(half, key=lambda k: half[k][si][h])
            picks.append(g)
            hon.append(half[g][si][1 - h] - half[base_key][si][1 - h])
    top = sorted(full, key=full.get)[:4]
    log(f"  [{name}] base {b0:.6f}; best on all H {best}: {full[best] - b0:+.7f}; top {[(k, round(full[k] - b0, 7)) for k in top]}")
    log(f"  [{name}] split-half honest mean {np.mean(hon):+.7f} (per split/half {np.round(hon, 7).tolist()}); picks {picks}")
    return dict(all_H=full[best] - b0, best=str(best), honest=float(np.mean(hon)), honest_list=[float(x) for x in hon],
                picks=[str(p) for p in picks], frac_neg=float(np.mean(np.array(hon) < 0)))


def main():
    logf = open("exp/audit/corrhyper/c01_shrink.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    tot = {}
    for m in ("gas", "eui", "elec"):
        F = z[f"{m}_F"].astype(np.float64)
        rP, pP, pH = z[f"{m}_rP"], z[f"{m}_pP"], z[f"{m}_teacher"]
        rH = z[f"{m}_y"] - pH
        sig = SIG[m]
        if m == "gas":
            num, den, mult = rP, A[P], A[H]
        elif m == "eui":
            num, den, mult = A[P] * rP, A[P], np.ones(len(H))
        else:
            num, den, mult = rP, pP, pH

        def G(Kz, Kc, Kw, WIN, w=None):
            return np.column_stack([group_mean(zone[P], zone[H], num, den, Kz, w) * mult,
                                    group_mean(cell[P], cell[H], num, den, Kc, w) * mult,
                                    window_mean(subn[P], subn[H], num, den, Kw, WIN, w) * mult])

        G0 = G(30, 30, 30, 30)
        err = np.abs(G0 - F[:, :3]).max() / np.abs(F[:, :3]).max()
        stored = np.sqrt(np.mean((z[f"{m}_y"] - z[f"{m}_corrected"]) ** 2)) / sig
        base = cv(F, rH, sig)
        log(f"== {m}: group recompute max rel err {err:.2e}; teacher {np.sqrt(np.mean(rH ** 2)) / sig:.6f}; "
            f"stored corrected {stored:.6f}; recomputed (float32 F) {base:.6f}")
        res = {"base": base, "stored": stored, "group_err": float(err)}
        Kn = F[:, 3:]
        cnts = [pool_count(c[1], c[2]) for c in CFG[m]]
        log(f"  kNN pool counts: share of rows with c<k per config {[round(float(np.mean(c < cf[2])), 4) for c, cf in zip(cnts, CFG[m])]}")

        def knn(Ks):
            return Kn * np.column_stack([(c / (c + Ks)) / (c / (c + 5.0)) for c in cnts])

        # E1 KSHR
        cands = {Ks: np.column_stack([F[:, :3], knn(Ks)]) for Ks in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)}
        res["E1_kshr"] = evaluate("E1 KSHR", cands, 5.0, rH, sig, log)
        # E2 two / three shrinkage levels for the group features
        cands = {"base": F}
        for lo in (3, 10):
            for hi in (100, 300):
                cands[(lo, hi)] = np.column_stack([G(lo, lo, lo, 30), G(hi, hi, hi, 30), Kn])
        cands[(3, 30, 300)] = np.column_stack([G(3, 3, 3, 30), G(30, 30, 30, 30), G(300, 300, 300, 30), Kn])
        cands[(10, 30)] = np.column_stack([G(10, 10, 10, 30), G0, Kn])
        cands[(30, 100)] = np.column_stack([G0, G(100, 100, 100, 30), Kn])
        res["E2_twolevel"] = evaluate("E2 two-level K", cands, "base", rH, sig, log)
        # E3 Kish effective-n shrinkage
        cands = {"base": F}
        for K in (3, 10, 30, 100):
            cands[("kish", K)] = np.column_stack([G(K, K, K, 30, "kish"), Kn])
        res["E3_kish"] = evaluate("E3 Kish n_eff", cands, "base", rH, sig, log)
        # E4 joint grid: common K for groups x WIN x KSHR
        cands = {}
        for K in (10, 30, 100):
            for WIN in (10, 30, 100):
                Gk = G(K, K, K, WIN)
                for Ks in (2.0, 5.0, 20.0):
                    cands[(K, WIN, Ks)] = np.column_stack([Gk, knn(Ks)])
        res["E4_joint"] = evaluate("E4 joint K x WIN x KSHR", cands, (30, 30, 5.0), rH, sig, log)
        OUT[m] = res
        log(f"  ({time.time() - t0:.0f}s)")
    for e in ("E1_kshr", "E2_twolevel", "E3_kish", "E4_joint"):
        tot[e] = {"honest_total": sum(OUT[m][e]["honest"] for m in OUT) / 5,
                  "allH_best_total": sum(OUT[m][e]["all_H"] for m in OUT) / 5}
        log(f"TOTAL {e}: honest split-half {tot[e]['honest_total']:+.8f}; all-H best (optimistic) {tot[e]['allH_best_total']:+.8f}")
    OUT["total"] = tot
    json.dump(OUT, open("exp/audit/corrhyper/c01_shrink.json", "w"), indent=1)
    log("wrote exp/audit/corrhyper/c01_shrink.json")


if __name__ == "__main__":
    main()
