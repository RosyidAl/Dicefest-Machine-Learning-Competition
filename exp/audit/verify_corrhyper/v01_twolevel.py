"""verify_corrhyper v01: independent re-check of corrhyper E2 (multi-level group shrinkage columns).
(a) reproduce E2 split-half numbers with the same 4 half seeds;
(b) 16 half splits (seeds 11..26) and a different inner KFold seed;
(c) nested outer-5-fold (select on 4/5 by inner 5-fold CV, refit lambda on the 4/5 (~103k rows, closer to the
    real full-H fit of ~129k rows), score on 1/5), 3 outer seeds;
(d) fixed pre-specified candidates (no selection), 5-fold and 10-fold cross-fitted on all H, extrapolated to
    n_fit = |H| via linear fit in 1/n_fit (lambda-estimation variance penalty removed as in the real pipeline).
All effects as change of TOTAL score (NRMSE change / 5)."""
import json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
tr = pd.read_parquet("cache/train.parquet", columns=["zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"])
zone = pd.factorize(tr["zone_region_id"])[0]
cell = pd.factorize(pd.Series(list(zip(tr["env_climate_01"], tr["env_climate_03"], tr["env_climate_05"]))))[0]
del tr
logf = open("exp/audit/verify_corrhyper/v01_twolevel.log", "w")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

def group_mean(gf, ga, num, den, K):
    ng = max(gf.max(), ga.max()) + 1
    s, dn, n = (np.bincount(gf, v, ng)[ga] for v in (num, den, np.ones(len(gf))))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)

def window_mean(sf, sa, num, den, K, WIN):
    m = max(sf.max(), sa.max()) + 1
    lo, hi = np.clip(sa - WIN, 0, m), np.clip(sa + WIN + 1, 0, m)
    def cs(v):
        c = np.r_[0, np.cumsum(np.bincount(sf, v, m))]; return c[hi] - c[lo]
    s, dn, n = cs(num), cs(den), cs(np.ones(len(sf)))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)

def cvpred(F, r, k=5, seed=0):
    pc = np.zeros(len(r))
    for a, b in KFold(k, shuffle=True, random_state=seed).split(F):
        pc[b] = F[b] @ np.linalg.lstsq(F[a], r[a], rcond=None)[0]
    return pc

def nr(r, pc, sig):
    return np.sqrt(np.mean((r - pc) ** 2)) / sig

OUT = {}
for m in ("gas", "eui", "elec"):
    F = z[f"{m}_F"].astype(np.float64)
    rP, pP, pH = z[f"{m}_rP"], z[f"{m}_pP"], z[f"{m}_teacher"]
    rH = z[f"{m}_y"] - pH; sig = SIG[m]
    if m == "gas": num, den, mult = rP, A[P], A[H]
    elif m == "eui": num, den, mult = A[P] * rP, A[P], np.ones(len(H))
    else: num, den, mult = rP, pP, pH
    G = lambda K: np.column_stack([group_mean(zone[P], zone[H], num, den, K) * mult,
                                   group_mean(cell[P], cell[H], num, den, K) * mult,
                                   window_mean(subn[P], subn[H], num, den, K, 30) * mult])
    G0 = G(30); Kn = F[:, 3:]
    log(f"== {m}: group recompute rel err {np.abs(G0 - F[:, :3]).max() / np.abs(F[:, :3]).max():.1e}; n_H {len(H)}; base cols {F.shape[1]}")
    C = {"base": F}
    for lo in (3, 10):
        for hi in (100, 300):
            C[f"{lo}/{hi}"] = np.column_stack([G(lo), G(hi), Kn])
    C["3/30/300"] = np.column_stack([G(3), G0, G(300), Kn])
    C["10/30"] = np.column_stack([G(10), G0, Kn])
    C["30/100"] = np.column_stack([G0, G(100), Kn])
    res = {}
    # (a)+(b) split-half
    for tag, seeds, ks in (("a_same4", range(11, 15), 0), ("b_16splits", range(11, 27), 0), ("b_16splits_kseed7", range(11, 27), 7)):
        hon, picks = [], []
        for s in seeds:
            hv = np.random.default_rng(s).permutation(len(H)) % 2
            sc = {k: [nr(rH[hv == h], cvpred(X[hv == h], rH[hv == h], 5, ks), sig) for h in (0, 1)] for k, X in C.items()}
            for h in (0, 1):
                g = min(sc, key=lambda k: sc[k][h]); picks.append(g)
                hon.append(sc[g][1 - h] - sc["base"][1 - h])
        res[tag] = dict(mean=float(np.mean(hon)), se=float(np.std(hon) / np.sqrt(len(hon))), npick_nonbase=int(sum(p != "base" for p in picks)))
        log(f"  [{tag}] split-half honest NRMSE {np.mean(hon):+.7f} (se {res[tag]['se']:.1e}); non-base picks {res[tag]['npick_nonbase']}/{len(picks)}")
    # (c) nested outer 5-fold
    for os_ in (1, 2, 3):
        pc_sel = np.zeros(len(H)); pc_base = np.zeros(len(H)); pk = []
        for a, b in KFold(5, shuffle=True, random_state=100 + os_).split(F):
            sc = {k: nr(rH[a], cvpred(X[a], rH[a], 5, 0), sig) for k, X in C.items()}
            g = min(sc, key=sc.get); pk.append(g)
            pc_sel[b] = C[g][b] @ np.linalg.lstsq(C[g][a], rH[a], rcond=None)[0]
            pc_base[b] = F[b] @ np.linalg.lstsq(F[a], rH[a], rcond=None)[0]
        d = nr(rH, pc_sel, sig) - nr(rH, pc_base, sig)
        res[f"c_nested_seed{os_}"] = dict(delta=float(d), picks=pk)
        log(f"  [c nested outer seed {os_}] honest NRMSE {d:+.7f}; picks {pk}")
    # (d) fixed candidates, extrapolate to n_fit = n_H
    n = len(H); fx = {}
    for k, X in C.items():
        v = {}
        for kf in (2, 5, 10):
            v[kf] = np.mean([nr(rH, cvpred(X, rH, kf, s), sig) for s in (0, 1)])
        fx[k] = v
    ext = {}
    for k in C:
        xs = np.array([1 / (n * (kf - 1) / kf) for kf in (2, 5, 10)])
        ys = np.array([fx[k][kf] for kf in (2, 5, 10)])
        b1, b0 = np.polyfit(xs, ys, 1)
        ext[k] = b0 + b1 / n  # fit size = all H
    for k in C:
        log(f"  [d fixed] {k:9s} cv5 {fx[k][5] - fx['base'][5]:+.7f}  cv10 {fx[k][10] - fx['base'][10]:+.7f}  extrap@n_H {ext[k] - ext['base']:+.7f}")
    res["d_fixed"] = {k: dict(cv5=fx[k][5] - fx["base"][5], cv10=fx[k][10] - fx["base"][10], ext=ext[k] - ext["base"]) for k in C}
    OUT[m] = res
tot = {}
for tag in ("a_same4", "b_16splits", "b_16splits_kseed7"):
    tot[tag] = sum(OUT[m][tag]["mean"] for m in OUT) / 5
for s in (1, 2, 3):
    tot[f"c_nested_seed{s}"] = sum(OUT[m][f"c_nested_seed{s}"]["delta"] for m in OUT) / 5
for k in OUT["gas"]["d_fixed"]:
    tot[f"d_fixed_ext_{k}"] = sum(OUT[m]["d_fixed"][k]["ext"] for m in OUT) / 5
tot["d_oracle_best_ext_per_target"] = sum(min(OUT[m]["d_fixed"][k]["ext"] for k in OUT[m]["d_fixed"]) for m in OUT) / 5
for k, v in tot.items():
    log(f"TOTAL {k}: {v:+.8f}")
OUT["total"] = tot
json.dump(OUT, open("exp/audit/verify_corrhyper/v01_twolevel.json", "w"), indent=1, default=str)
