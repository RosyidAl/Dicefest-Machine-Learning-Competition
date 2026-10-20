"""Independent re-check (verify2_postproc, round 2) of claim S1: rows with few meters (bld_dim_12 in {2,3} and 4)
are under-predicted. Uses only exp/audit/chain_H.npz (honest holdout chain).

1. multiplicative LS bias d = sum(p r)/sum(p^2) by M group on H (teacher, corrected) and on P (P-OOF teacher)
2. honest gains of p *= 1 + d_g on H corrected:
   a. deltas from P (the claim's own protocol, recomputed with this code)
   b. deltas cross-fitted inside H (5-fold and 2-fold, many seeds), from corrected or teacher residuals
   c. deltas from P plus the other H folds (pooled)
   d. reverse direction: deltas from H corrected, scored on P (P-OOF teacher)
3. sd of the gain from the per-row contributions (rows resampled, deltas held fixed)
All effects are reported per metric (NRMSE) and as change of the TOTAL score (= sum over metrics / 5).

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/r2_smallM/s01_H.py
"""
import json

import numpy as np

SIG = dict(gas=151750.04, eui=40.65, elec=41725.37)
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify2_postproc/r2_smallM"

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, Mall = z["P"], z["H"], z["M"]
MH, MP = Mall[H], Mall[P]
res = {}


def grp(M):
    """group label per row: 'g23' for M in {2,3}, 'g4' for M == 4, '' otherwise (M == 1 never occurs in train)"""
    g = np.full(len(M), "", dtype=object)
    g[(M == 2) | (M == 3)] = "g23"
    g[M == 4] = "g4"
    return g


GH, GP = grp(MH), grp(MP)
GROUPS = ("g23", "g4")


def ls_bias(p, r):
    """multiplicative LS shift and HC0 standard error"""
    s2 = float(np.sum(p * p))
    d = float(np.sum(p * r)) / s2
    se = float(np.sqrt(np.sum(p * p * (r - d * p) ** 2))) / s2
    return d, se


def deltas(p, r, g):
    return {k: ls_bias(p[g == k], r[g == k])[0] for k in GROUPS}


def apply(p, g, dl):
    q = p.copy()
    for k in GROUPS:
        q[g == k] *= 1.0 + dl[k]
    return q


def dnrmse(y, p0, p1, sig):
    return (np.sqrt(np.mean((y - p1) ** 2)) - np.sqrt(np.mean((y - p0) ** 2))) / sig


def gain_sd(y, p0, p1, sig, nboot=2000, seed=0):
    """sd of the NRMSE change when rows are resampled (deltas fixed): bootstrap on per-row squared-error terms"""
    rng = np.random.default_rng(seed)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    n = len(y)
    out = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, n, n)
        out[b] = (np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())) / sig
    return float(out.std())


# 0. counts
cnt = {int(m): int((Mall == m).sum()) for m in range(1, 9)}
print("train rows by M (1..8):", cnt)
print("H rows: g23", int((GH == "g23").sum()), "g4", int((GH == "g4").sum()),
      "| P rows: g23", int((GP == "g23").sum()), "g4", int((GP == "g4").sum()))
res["counts"] = cnt

# 1. bias tables
print("\n1. multiplicative LS bias d (se) by M")
bias = {}
for m in MET:
    y, t, c = z[f"{m}_y"], z[f"{m}_teacher"], z[f"{m}_corrected"]
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    rows = []
    for lab, selH, selP in [("M=2", MH == 2, MP == 2), ("M=3", MH == 3, MP == 3), ("M2-3", GH == "g23", GP == "g23"),
                            ("M=4", MH == 4, MP == 4), ("M=5", MH == 5, MP == 5), ("M=6", MH == 6, MP == 6),
                            ("M7-10", (MH >= 7) & (MH <= 10), (MP >= 7) & (MP <= 10)), ("M>10", MH > 10, MP > 10)]:
        dt = ls_bias(t[selH], y[selH] - t[selH]) if selH.sum() > 1 else (np.nan, np.nan)
        dc = ls_bias(c[selH], y[selH] - c[selH]) if selH.sum() > 1 else (np.nan, np.nan)
        dp = ls_bias(pP[selP], rP[selP]) if selP.sum() > 1 else (np.nan, np.nan)
        rows.append(dict(group=lab, nH=int(selH.sum()), nP=int(selP.sum()), H_teacher=dt, H_corr=dc, P_teacher=dp))
        print(f"  {m:4s} {lab:6s} nH {selH.sum():6d} nP {selP.sum():6d} | H teacher {dt[0]:+.3f} ({dt[1]:.3f}) "
              f"H corr {dc[0]:+.3f} ({dc[1]:.3f}) | P teacher {dp[0]:+.3f} ({dp[1]:.3f})")
    bias[m] = rows
res["bias"] = bias

# 2a. deltas from P -> H corrected (claim protocol)
print("\n2a. deltas from P-OOF teacher, applied to H corrected (claim protocol)")
g2a, tot = {}, 0.0
for m in MET:
    y, c = z[f"{m}_y"], z[f"{m}_corrected"]
    dl = deltas(z[f"{m}_pP"], z[f"{m}_rP"], GP)
    c1 = apply(c, GH, dl)
    dn = dnrmse(y, c, c1, SIG[m])
    sd = gain_sd(y, c, c1, SIG[m])
    # split of the gain into the two groups
    part = {k: dnrmse(y, c, apply(c, GH, {kk: (dl[kk] if kk == k else 0.0) for kk in GROUPS}), SIG[m]) for k in GROUPS}
    g2a[m] = dict(deltas=dl, dnrmse=dn, sd=sd, by_group=part)
    tot += dn / 5
    print(f"  {m:4s} d23 {dl['g23']:+.4f} d4 {dl['g4']:+.4f} | dNRMSE {dn:+.7f} (sd {sd:.7f}) "
          f"g23 {part['g23']:+.7f} g4 {part['g4']:+.7f}")
print(f"  TOTAL change {tot:+.7f}")
res["2a_P_to_H"] = dict(per_metric=g2a, total=tot)


# 2b/2c. cross-fitting inside H
def crossfit(m, K, seeds, source="corr", pool_P=False):
    y, t, c = z[f"{m}_y"], z[f"{m}_teacher"], z[f"{m}_corrected"]
    base = c if source == "corr" else t
    r = y - base
    out = []
    for s in seeds:
        fold = np.random.default_rng(1000 + s).integers(0, K, len(y))
        c1 = c.copy()
        for f in range(K):
            tr_ = fold != f
            if pool_P:
                pp = np.concatenate([z[f"{m}_pP"], base[tr_]])
                rr = np.concatenate([z[f"{m}_rP"], r[tr_]])
                gg = np.concatenate([GP, GH[tr_]])
                dl = deltas(pp, rr, gg)
            else:
                dl = deltas(base[tr_], r[tr_], GH[tr_])
            te_ = ~tr_
            c1[te_] = apply(c[te_], GH[te_], dl)
        out.append(dnrmse(y, c, c1, SIG[m]))
    return np.array(out)


print("\n2b/2c. deltas cross-fitted inside H (random fold assignment per seed)")
cf = {}
for lab, K, src, pool in [("K5_corr", 5, "corr", False), ("K2_corr", 2, "corr", False),
                          ("K5_teacher", 5, "teacher", False), ("K5_corr_poolP", 5, "corr", True)]:
    seeds = range(40) if not pool else range(10)
    per = {m: crossfit(m, K, seeds, src, pool) for m in MET}
    tot_s = sum(per[m] for m in MET) / 5
    cf[lab] = {m: dict(mean=float(per[m].mean()), sd_seeds=float(per[m].std())) for m in MET}
    cf[lab]["total"] = dict(mean=float(tot_s.mean()), sd_seeds=float(tot_s.std()), min=float(tot_s.min()),
                            max=float(tot_s.max()), frac_neg=float((tot_s < 0).mean()))
    print(f"  {lab:14s} " + " ".join(f"{m} {per[m].mean():+.7f}" for m in MET)
          + f" | TOTAL {tot_s.mean():+.7f} (seed sd {tot_s.std():.7f}, range {tot_s.min():+.7f}..{tot_s.max():+.7f},"
          f" frac<0 {np.mean(tot_s < 0):.2f})")
res["2bc_crossfit_H"] = cf

# 2d. reverse: deltas from H corrected -> P teacher
print("\n2d. reverse: deltas from H corrected, scored on P (P-OOF teacher)")
g2d, tot = {}, 0.0
for m in MET:
    y_c = z[f"{m}_y"] - z[f"{m}_corrected"]
    dl = deltas(z[f"{m}_corrected"], y_c, GH)
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    yP = pP + rP
    p1 = apply(pP, GP, dl)
    dn = dnrmse(yP, pP, p1, SIG[m])
    sd = gain_sd(yP, pP, p1, SIG[m], nboot=1000)
    g2d[m] = dict(deltas=dl, dnrmse=dn, sd=sd)
    tot += dn / 5
    print(f"  {m:4s} d23 {dl['g23']:+.4f} d4 {dl['g4']:+.4f} | dNRMSE on P {dn:+.7f} (sd {sd:.7f})")
print(f"  TOTAL change {tot:+.7f}")
res["2d_H_to_P"] = dict(per_metric=g2d, total=tot)

# 3. additive form under the claim protocol, for comparison
print("\n3. additive form (shift = mean residual of the group, from P), applied to H corrected")
tot = 0.0
for m in MET:
    y, c = z[f"{m}_y"], z[f"{m}_corrected"]
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    c1 = c.copy()
    for k in GROUPS:
        c1[GH == k] += rP[GP == k].mean()
    dn = dnrmse(y, c, c1, SIG[m])
    tot += dn / 5
    print(f"  {m:4s} dNRMSE {dn:+.7f}")
print(f"  TOTAL change {tot:+.7f}")
res["3_additive_total"] = tot

json.dump(res, open(f"{OUT}/s01_H.json", "w"), indent=1, default=float)
print("\nwrote", f"{OUT}/s01_H.json")
