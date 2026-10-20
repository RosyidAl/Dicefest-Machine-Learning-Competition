"""Re-tune the group-correction constants K (shrinkage n/(n+K)) and WIN (sub_n half-window), fixed at 30/30 since
v9 (14 Oct) and never re-tuned for the VC blend + kNN correction. Honest: lambda cross-fitted 5-fold on H, kNN columns
unchanged (chain_H.npz F[:, 3:]); choice of K/WIN checked by split-half (choose on one half, score on the other)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
tr = pd.read_parquet("cache/train.parquet", columns=["zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"])
zone = pd.factorize(tr["zone_region_id"])[0]
cell = pd.factorize(pd.Series(list(zip(tr["env_climate_01"], tr["env_climate_03"], tr["env_climate_05"]))))[0]
del tr


def group_mean(gf, ga, num, den, K):
    ng = max(gf.max(), ga.max()) + 1
    s, dn, n = (np.bincount(gf, v, ng)[ga] for v in (num, den, np.ones(len(gf))))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def window_mean(sf, sa, num, den, K, WIN):
    m = max(sf.max(), sa.max()) + 1
    lo, hi = np.clip(sa - WIN, 0, m), np.clip(sa + WIN + 1, 0, m)
    s, dn, n = ((c := np.r_[0, np.cumsum(np.bincount(sf, v, m))])[hi] - c[lo] for v in (num, den, np.ones(len(sf))))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def cv(F, r, sig, idx=None):
    if idx is not None:
        F, r = F[idx], r[idx]
    pc = np.zeros(len(r))
    for a, b in KFold(5, shuffle=True, random_state=0).split(F):
        pc[b] = F[b] @ np.linalg.lstsq(F[a], r[a], rcond=None)[0]
    return np.sqrt(np.mean((r - pc) ** 2)) / sig


halves = np.random.default_rng(11).permutation(len(H)) % 2
tot = 0.0
for m in ("gas", "eui", "elec"):
    F = z[f"{m}_F"].astype(np.float64)
    rP, pP = z[f"{m}_rP"], z[f"{m}_pP"]
    rH = z[f"{m}_y"] - z[f"{m}_teacher"]
    pH = z[f"{m}_teacher"]
    if m == "gas":
        num, den, mult = rP, A[P], A[H]
    elif m == "eui":
        num, den, mult = A[P] * rP, A[P], np.ones(len(H))
    else:
        num, den, mult = rP, pP, pH

    def cols(Kz, Kc, Kw, WIN):
        return np.column_stack([group_mean(zone[P], zone[H], num, den, Kz) * mult,
                                group_mean(cell[P], cell[H], num, den, Kc) * mult,
                                window_mean(subn[P], subn[H], num, den, Kw, WIN) * mult])

    G0 = cols(30, 30, 30, 30)
    err = np.abs(G0 - F[:, :3]).max() / np.abs(F[:, :3]).max()
    print(f"== {m}: recomputed group features vs stored, max rel err {err:.2e}", flush=True)
    base = cv(F, rH, SIG[m])
    print(f"   base (K=30 x3, WIN=30) {base:.6f}", flush=True)
    grid = []
    for Kz in (10, 30, 100):
        for Kc in (3, 10, 30, 100):
            for Kw in (10, 30, 100):
                for WIN in (10, 20, 30, 50):
                    grid.append((Kz, Kc, Kw, WIN))
    scores = {}
    for g in grid:
        Fg = np.column_stack([cols(*g), F[:, 3:]])
        scores[g] = (cv(Fg, rH, SIG[m]), [cv(Fg, rH, SIG[m], halves == h) for h in (0, 1)])
    best = min(scores, key=lambda g: scores[g][0])
    print(f"   best on all H {best}: {scores[best][0]:.6f} ({scores[best][0] - base:+.7f})", flush=True)
    top = sorted(scores, key=lambda g: scores[g][0])[:5]
    print("   top5:", [(g, round(scores[g][0] - base, 7)) for g in top], flush=True)
    # split-half honest: choose on half h, score on the other half (vs base on that half)
    base_h = [scores[(30, 30, 30, 30)][1][h] for h in (0, 1)]
    hon = []
    for h in (0, 1):
        g = min(scores, key=lambda g: scores[g][1][h])
        hon.append(scores[g][1][1 - h] - base_h[1 - h])
        print(f"   choose on half {h}: {g} -> other half {hon[-1]:+.7f}", flush=True)
    print(f"   honest split-half mean {np.mean(hon):+.7f}  total {np.mean(hon) / 5:+.8f}", flush=True)
    tot += np.mean(hon) / 5
    # extra windows as additional columns
    Fx = np.column_stack([F, window_mean(subn[P], subn[H], num, den, 30, 10) * mult,
                          window_mean(subn[P], subn[H], num, den, 30, 100) * mult])
    ex = cv(Fx, rH, SIG[m])
    hx = [cv(Fx, rH, SIG[m], halves == h) - cv(F, rH, SIG[m], halves == h) for h in (0, 1)]
    print(f"   + windows 10 and 100 as extra columns: {ex:.6f} ({ex - base:+.7f}); per half {np.round(hx, 7)}", flush=True)
print(f"TOTAL honest split-half change of re-tuned K/WIN: {tot:+.8f}")
