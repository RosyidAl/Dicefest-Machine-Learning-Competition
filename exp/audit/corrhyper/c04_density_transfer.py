"""corrhyper c04: does the group-shrinkage K transfer across fit-pool density (holdout P -> test uses all train rows,
1/0.7 denser)? Simulation one level down: P' = random 70% of P. lambda is fitted (5-fold cross-fitted) on H with the
group columns computed from P' residuals, then applied to H with group columns computed from all of P (the denser
pool, as at test time). Compare with lambda fitted directly on the dense-pool columns (cross-fitted). kNN columns are
kept as stored (from all of P) in both, so only the group part moves. Gap = transferred - direct; small gap for a K
means the H-fitted lambda is right for the denser test pool. Also: gain of the dense-pool correction under each K.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
tr = pd.read_parquet("cache/train.parquet", columns=["zone_region_id", "env_climate_01", "env_climate_03",
                                                     "env_climate_05"])
zone = pd.factorize(tr["zone_region_id"])[0]
cell = pd.factorize(pd.Series(list(zip(tr["env_climate_01"], tr["env_climate_03"], tr["env_climate_05"]))))[0]
del tr
logf = open("exp/audit/corrhyper/c04_density_transfer.log", "w")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def group_mean(gf, ga, num, den, K):
    ng = max(gf.max(), ga.max()) + 1
    s, dn, n = (np.bincount(gf, v, ng)[ga] for v in (num, den, np.ones(len(gf))))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def window_mean(sf, sa, num, den, K, WIN):
    m = max(sf.max(), sa.max()) + 1
    lo, hi = np.clip(sa - WIN, 0, m), np.clip(sa + WIN + 1, 0, m)
    s, dn, n = ((c := np.r_[0, np.cumsum(np.bincount(sf, v, m))])[hi] - c[lo] for v in (num, den, np.ones(len(sf))))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def rmse(r, sig):
    return np.sqrt(np.mean(r ** 2)) / sig


OUT = {}
folds = list(KFold(5, shuffle=True, random_state=0).split(H))
for m in ("gas", "eui", "elec"):
    F = z[f"{m}_F"].astype(np.float64)
    rP, pP, pH = z[f"{m}_rP"], z[f"{m}_pP"], z[f"{m}_teacher"]
    rH = z[f"{m}_y"] - pH
    sig = SIG[m]
    Kn = F[:, 3:]

    def G(sel, K, WIN=30):
        idx = P[sel]
        if m == "gas":
            num, den, mult = rP[sel], A[idx], A[H]
        elif m == "eui":
            num, den, mult = A[idx] * rP[sel], A[idx], np.ones(len(H))
        else:
            num, den, mult = rP[sel], pP[sel], pH
        return np.column_stack([group_mean(zone[idx], zone[H], num, den, K),
                                group_mean(cell[idx], cell[H], num, den, K),
                                window_mean(subn[idx], subn[H], num, den, K, WIN)]) * np.reshape(mult, (-1, 1))

    allsel = np.ones(len(P), bool)
    OUT[m] = {}
    for K in (10, 30, 100):
        Fd = np.column_stack([G(allsel, K), Kn])
        direct = np.zeros(len(H))
        for a, b in folds:
            direct[b] = Fd[b] @ np.linalg.lstsq(Fd[a], rH[a], rcond=None)[0]
        gaps, lamr = [], []
        for seed in (0, 1, 2):
            sub = np.random.default_rng(100 + seed).random(len(P)) < 0.7
            Fs = np.column_stack([G(sub, K), Kn])
            tf = np.zeros(len(H))
            for a, b in folds:
                lam = np.linalg.lstsq(Fs[a], rH[a], rcond=None)[0]
                tf[b] = Fd[b] @ lam
            gaps.append(rmse(rH - tf, sig) - rmse(rH - direct, sig))
            ls = np.linalg.lstsq(Fs, rH, rcond=None)[0][:3]
            ld = np.linalg.lstsq(Fd, rH, rcond=None)[0][:3]
            lamr.append((ls / ld).tolist())
        log(f"{m} K={K:3d}: direct dense {rmse(rH - direct, sig):.6f}; transferred-from-sparse gap "
            f"{np.mean(gaps):+.7f} (seeds {np.round(gaps, 7).tolist()}); lambda sparse/dense group cols "
            f"{np.round(np.mean(lamr, 0), 3).tolist()}")
        OUT[m][K] = dict(direct=float(rmse(rH - direct, sig)), gap=float(np.mean(gaps)), lam_ratio=np.mean(lamr, 0).tolist())
json.dump(OUT, open("exp/audit/corrhyper/c04_density_transfer.json", "w"), indent=1)
