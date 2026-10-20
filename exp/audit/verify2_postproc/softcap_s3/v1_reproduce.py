"""verify2_postproc v1: independent reproduction of claim S3 (truncated-normal soft cap near the gas 1e6 truncation).

Rule under test (gas and eui, upper bound only):
    p' = p + k * adj,  adj = -s * phi(b) / Phi(b),  b = (T - p) / s,  s = p * sqrt(a0 + b0 / M)
    T = T_gas for gas, T_gas / A for eui.

Honest designs (all scored on H = 30% holdout, corrected chain prediction):
  D0  bias of residual y - p by standardized distance to the cap, on H (corrected and teacher) and on P (teacher OOF).
  D1  k fitted on P only (teacher P-OOF blend, 300k rows, H never seen) and applied to H corrected.
  D2  fixed k (1.0 = pure truncated-normal theory, 0.5) - no fitting of k at all.
  D3  k cross-fitted on H with many random partitions (2-fold x 200 seeds, 5-fold x 50, 10-fold x 20), seeds
      different from the auditor's (KFold random_state=0).
  D4  bootstrap of H rows (2000 resamples) for the D1 rule: sampling uncertainty of the gain.
Cap variants: T = 1e6 (round threshold), T = max(y_P) (P-derived, honest for H), T = 999764.8988 (claimed, full train).
(a0, b0): own OLS of (r/p)^2 on [1, 1/M] over P rows; sensitivity with a log-residual version.
Every effect is reported as change of the TOTAL score = per-target NRMSE change / 5.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/softcap_s3/v1_reproduce.py
"""
import json
import sys

import numpy as np
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS  # noqa: E402

SIG = {"gas": SIGMAS["target_macro_gas_kwh"], "eui": SIGMAS["target_eui_gas_kwh_per_sqm"]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
out = {}


def mills_adj(p, s, T):
    """E[Y | Y < T] - p for Y ~ N(p, s^2): -s * phi(b) / Phi(b), computed in log space for stability."""
    b = (T - p) / s
    logphi = -0.5 * b * b - 0.5 * np.log(2 * np.pi)
    return -s * np.exp(logphi - log_ndtr(b))


def nrmse(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def ls_k(adj, r):
    den = float(adj @ adj)
    return float(adj @ r / den) if den > 0 else 0.0


# ---------- P side (teacher P-OOF blend) ----------
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in SIG}
TP_max = float(yP["gas"].max())
caps = {"T=1e6": 1e6, "T=maxP": TP_max, "T=claimed": 999764.8988}
print(f"P rows {len(P)}  H rows {len(H)}  max gas P {TP_max:.1f}  max gas H {z['gas_y'].max():.1f}")
# consistency checks: eui*A == gas on P and H
print("check gas == eui*A on P:", float(np.max(np.abs(yP['gas'] - yP['eui'] * AP) / yP['gas'])),
      " on H:", float(np.max(np.abs(z['gas_y'] - z['eui_y'] * AH) / z['gas_y'])))

# variance model fitted on P only
VM = {}
for m in SIG:
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    X = np.column_stack([np.ones(len(p)), 1.0 / MP])
    ols = np.linalg.lstsq(X, (r / p) ** 2, rcond=None)[0]
    lg = np.linalg.lstsq(X, np.log(np.clip(yP[m], 1e-9, None) / p) ** 2, rcond=None)[0]
    VM[m] = {"ols_rel": ols.tolist(), "ols_log": lg.tolist()}
    print(f"{m}: var model (r/p)^2 = {ols[0]:.5f} + {ols[1]:.5f}/M ; log-resid^2 = {lg[0]:.5f} + {lg[1]:.5f}/M")
out["var_model"] = VM


def sd(m, p, Mv, kind="ols_rel"):
    a0, b0 = VM[m][kind]
    return p * np.sqrt(np.clip(a0 + b0 / Mv, 1e-8, None))


# ---------- D0: bias tables ----------
bands = [(-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)]


def bias_table(tag, m, y, p, Mv, Av, T):
    s = sd(m, p, Mv)
    Tm = T if m == "gas" else T / Av
    dist = (Tm - p) / s
    r = y - p
    adj = mills_adj(p, s, Tm)
    rows = []
    print(f"  [{tag}] {m}")
    for lo, hi in bands:
        sel = (dist >= lo) & (dist < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        mr, se = float(r[sel].mean()), float(r[sel].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append(dict(band=[lo, hi], n=n, mean_resid=mr, se=se, t=mr / se if se == se and se > 0 else None,
                         mean_resid_over_s=float(np.mean(r[sel] / s[sel])), mean_adj_over_s=float(np.mean(adj[sel] / s[sel]))))
        print(f"    dist [{lo:>4},{hi:>5}) n {n:6d} mean resid {mr:+12.3f} se {se:10.3f} t {mr / se if se > 0 else 0:+5.1f}"
              f"  resid/s {rows[-1]['mean_resid_over_s']:+.3f}  tn-adj/s {rows[-1]['mean_adj_over_s']:+.3f}")
    return rows


T0 = caps["T=maxP"]
out["D0"] = {}
for m in SIG:
    out["D0"][f"H_corrected_{m}"] = bias_table("H corrected", m, z[f"{m}_y"], z[f"{m}_corrected"], MH, AH, T0)
    out["D0"][f"H_teacher_{m}"] = bias_table("H teacher", m, z[f"{m}_y"], z[f"{m}_teacher"], MH, AH, T0)
    out["D0"][f"P_teacher_{m}"] = bias_table("P teacher OOF", m, yP[m], z[f"{m}_pP"], MP, AP, T0)

# top-quantile calibration on H corrected and P teacher (not dependent on s model)
print("  top-quantile calibration (mean residual of rows with the largest predictions)")
for m in SIG:
    for tag, y, p in (("H corr", z[f"{m}_y"], z[f"{m}_corrected"]), ("P teach", yP[m], z[f"{m}_pP"])):
        o = np.argsort(p)
        for q0, q1 in ((0.99, 0.999), (0.999, 1.0)):
            idx = o[int(q0 * len(p)):int(q1 * len(p))]
            r = y[idx] - p[idx]
            print(f"    {m:4s} {tag:8s} q[{q0},{q1}] n {len(idx):5d} mean p {p[idx].mean():12.2f} mean resid "
                  f"{r.mean():+11.3f} t {r.mean() / (r.std(ddof=1) / np.sqrt(len(idx))):+5.1f}")

# ---------- D1/D2: k from P, fixed k ----------
res = {}
for cname, T in caps.items():
    res[cname] = {}
    tot = {"kP": 0.0, "kP_clip": 0.0, "k1": 0.0, "k05": 0.0, "clip_only": 0.0, "kP_teacherH": 0.0}
    for m in SIG:
        # P fit
        pPm, ym = z[f"{m}_pP"], yP[m]
        TmP = T if m == "gas" else T / AP
        adjP = mills_adj(pPm, sd(m, pPm, MP), TmP)
        kP = ls_k(adjP, ym - pPm)
        # H apply
        y, p = z[f"{m}_y"], z[f"{m}_corrected"]
        TmH = T if m == "gas" else T / AH
        adjH = mills_adj(p, sd(m, p, MH), TmH)
        base = nrmse(y, p, m)
        d = {}
        d["kP"] = nrmse(y, p + kP * adjH, m) - base
        d["kP_clip"] = nrmse(y, np.minimum(p + kP * adjH, TmH), m) - base
        d["k1"] = nrmse(y, p + 1.0 * adjH, m) - base
        d["k05"] = nrmse(y, p + 0.5 * adjH, m) - base
        d["clip_only"] = nrmse(y, np.minimum(p, TmH), m) - base
        # teacher on H with the same P-fitted k (rule applied before correction would act like this)
        pt = z[f"{m}_teacher"]
        adjT = mills_adj(pt, sd(m, pt, MH), TmH)
        d["kP_teacherH"] = nrmse(y, pt + kP * adjT, m) - nrmse(y, pt, m)
        # in-sample k on H (reference only, NOT honest)
        kH_in = ls_k(adjH, y - p)
        res[cname][m] = dict(kP=kP, kH_insample=kH_in, base=base, n_adj_gt_1e3sig=int((np.abs(kP * adjH) > 1e-3 * SIG[m]).sum()),
                             n_adj_gt_1e2sig=int((np.abs(kP * adjH) > 1e-2 * SIG[m]).sum()),
                             n_adj_gt_1e1sig=int((np.abs(kP * adjH) > 1e-1 * SIG[m]).sum()), dNRMSE=d)
        for key in tot:
            tot[key] += d[key] / 5
        print(f"  {cname:10s} {m:4s} kP {kP:.4f} (H in-sample k {kH_in:.4f}) base {base:.6f} | "
              + " ".join(f"{key} {v:+.7f}" for key, v in d.items()))
    res[cname]["TOTAL"] = tot
    print(f"  {cname:10s} TOTAL " + " ".join(f"{key} {v:+.8f}" for key, v in tot.items()))
out["D1_D2"] = res

# ---------- D3: cross-fitted k on H, many partitions ----------
T = caps["T=maxP"]
pre = {}
for m in SIG:
    y, p = z[f"{m}_y"], z[f"{m}_corrected"]
    TmH = T if m == "gas" else T / AH
    pre[m] = (y, p, mills_adj(p, sd(m, p, MH), TmH), nrmse(y, p, m))
d3 = {}
for nf, nseeds in ((2, 200), (5, 50), (10, 20)):
    tots, per = [], {m: [] for m in SIG}
    for seed in range(101, 101 + nseeds):
        tot = 0.0
        for m in SIG:
            y, p, adj, base = pre[m]
            pc = p.copy()
            for a, b in KFold(nf, shuffle=True, random_state=seed).split(p):
                pc[b] = p[b] + ls_k(adj[a], (y - p)[a]) * adj[b]
            dm = nrmse(y, pc, m) - base
            per[m].append(dm)
            tot += dm / 5
        tots.append(tot)
    tots = np.array(tots)
    d3[f"{nf}fold"] = dict(n_seeds=nseeds, total_mean=float(tots.mean()), total_sd=float(tots.std()),
                           frac_neg=float((tots < 0).mean()),
                           per_metric_mean={m: float(np.mean(v)) for m, v in per.items()})
    print(f"  D3 {nf:2d}-fold x {nseeds:3d} seeds: total mean {tots.mean():+.8f} sd {tots.std():.8f} "
          f"frac<0 {(tots < 0).mean():.2f} | " + " ".join(f"{m} {np.mean(v):+.7f}" for m, v in per.items()))
out["D3"] = d3

# split-half: fit on one half, score on the OTHER half only (pure out-of-sample on each half)
sh = []
for seed in range(1001, 1201):
    h = np.random.default_rng(seed).permutation(len(H)) % 2
    tot = 0.0
    for m in SIG:
        y, p, adj, _ = pre[m]
        a, b = h == 0, h == 1
        k = ls_k(adj[a], (y - p)[a])
        tot += (nrmse(y[b], p[b] + k * adj[b], m) - nrmse(y[b], p[b], m)) / 5
    sh.append(tot)
sh = np.array(sh)
out["D3_halfscore"] = dict(mean=float(sh.mean()), sd=float(sh.std()), frac_neg=float((sh < 0).mean()))
print(f"  D3 fit-half/score-other-half x200: total mean {sh.mean():+.8f} sd {sh.std():.8f} frac<0 {(sh < 0).mean():.2f}")

# ---------- D4: bootstrap of the P-fitted rule on H ----------
kPs = {m: res["T=maxP"][m]["kP"] for m in SIG}
rng = np.random.default_rng(2024)
B = 2000
boot = np.zeros(B)
n = len(H)
sq = {}
for m in SIG:
    y, p, adj, _ = pre[m]
    sq[m] = ((y - p) ** 2, (y - p - kPs[m] * adj) ** 2)
for i in range(B):
    idx = rng.integers(0, n, n)
    tot = 0.0
    for m in SIG:
        e0, e1 = sq[m]
        tot += (np.sqrt(e1[idx].mean()) - np.sqrt(e0[idx].mean())) / SIG[m] / 5
    boot[i] = tot
out["D4_bootstrap_kP"] = dict(mean=float(boot.mean()), sd=float(boot.std()), q025=float(np.quantile(boot, 0.025)),
                              q975=float(np.quantile(boot, 0.975)), frac_neg=float((boot < 0).mean()))
print(f"  D4 bootstrap (k from P, T=maxP): total mean {boot.mean():+.8f} sd {boot.std():.8f} "
      f"95% [{np.quantile(boot, 0.025):+.8f}, {np.quantile(boot, 0.975):+.8f}] frac<0 {(boot < 0).mean():.3f}")

# sensitivity to the variance model (log-residual version), k from P
tot = 0.0
for m in SIG:
    pPm, ym = z[f"{m}_pP"], yP[m]
    kP = ls_k(mills_adj(pPm, sd(m, pPm, MP, "ols_log"), T if m == "gas" else T / AP), ym - pPm)
    y, p = z[f"{m}_y"], z[f"{m}_corrected"]
    adjH = mills_adj(p, sd(m, p, MH, "ols_log"), T if m == "gas" else T / AH)
    dm = nrmse(y, p + kP * adjH, m) - nrmse(y, p, m)
    tot += dm / 5
    print(f"  var-model log version {m}: kP {kP:.4f} d {dm:+.7f}")
out["sens_logvar_total"] = tot
print(f"  var-model log version TOTAL {tot:+.8f}")

json.dump(out, open("exp/audit/verify2_postproc/softcap_s3/v1_reproduce.json", "w"), indent=1)
print("wrote exp/audit/verify2_postproc/softcap_s3/v1_reproduce.json")
