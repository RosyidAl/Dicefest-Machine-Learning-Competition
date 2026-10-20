"""verify2_postproc/combo_repro c1: independent re-implementation (own code) of the combined post-processing chain
S1 (small-M multiplicative shift) + S3 (soft cap below the gas truncation) + S4 (hard clip to train-implied bounds)
on the honest holdout predictions in exp/audit/chain_H.npz, scored with several honest splits:

  (a) the claim's protocol (S1 deltas and S3 noise model and S4 bounds from P; S3 slope k cross-fitted 5-fold on H,
      KFold seed 0)                                                     -> must reproduce about -6.5e-05 total
  (b) other cross-fitting splits for k: 5-fold seeds 1..100, 2-fold (split-half) seeds 1..100, 10-fold seeds 1..20
  (c) paired bootstrap / delta-method SE of the total change on H, per-step and per-metric decomposition,
      concentration of the gain in a few rows
  (d) reverse direction for S1: deltas fitted on H (corrected residuals), applied to the P-side teacher, scored on P
  (e) S1 estimator / group variants, S3 variants (noise model from H fit folds, stable tail formula)

All effects are reported as change of the TOTAL score (= sum over the 3 metrics of the NRMSE change / 5).
usage (from /root/Dicefest): OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
    exp/audit/verify2_postproc/combo_repro/c1_chainH.py
"""
import json

import numpy as np
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

OUT = "exp/audit/verify2_postproc/combo_repro/c1_chainH.json"
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
MET = ("gas", "eui", "elec")
GROUPS = ((2, 3), (4, 4))

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"].astype(float), z["M"].astype(float)
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
nH = len(H)
y = {m: z[f"{m}_y"].astype(float) for m in MET}
base = {m: np.clip(z[f"{m}_corrected"].astype(float), 0, None) for m in MET}
pP = {m: z[f"{m}_pP"].astype(float) for m in MET}
rP = {m: z[f"{m}_rP"].astype(float) for m in MET}
yP = {m: pP[m] + rP[m] for m in MET}
res = {}


def nrmse(m, p, idx=None):
    if idx is None:
        return float(np.sqrt(np.mean((y[m] - p) ** 2)) / SIG[m])
    return float(np.sqrt(np.mean((y[m][idx] - p[idx]) ** 2)) / SIG[m])


base_sc = {m: nrmse(m, base[m]) for m in MET}
print("base NRMSE on H", {m: round(v, 6) for m, v in base_sc.items()})
res["base"] = base_sc

# ---- parameters that come from P only -------------------------------------------------------------------------
B = dict(gmax=float(yP["gas"].max()), euimin=float(yP["eui"].min()),
         eamin=float((yP["elec"] / AP).min()), eamax=float((yP["elec"] / AP).max()))
print("bounds from P", B)


def ls_delta(p, r):
    return float(np.dot(p, r) / np.dot(p, p))


def deltas_from(pf, rf, Mf, est="ls"):
    out = {}
    for m in MET:
        out[m] = []
        for lo, hi in GROUPS:
            g = (Mf >= lo) & (Mf <= hi)
            p, r = pf[m][g], rf[m][g]
            if est == "ls":
                out[m].append(ls_delta(p, r))
            elif est == "ros":
                out[m].append(float(r.sum() / p.sum()))
            elif est == "meanrel":
                out[m].append(float(np.mean(r / p)))
    return out


DP = deltas_from(pP, rP, MP)
print("S1 deltas from P (LS)", {m: [round(v, 4) for v in d] for m, d in DP.items()})
res["deltas_P"] = DP


def noise_model(p, r, Mv):
    X = np.column_stack([np.ones(len(p)), 1.0 / Mv])
    return np.linalg.lstsq(X, (r / p) ** 2, rcond=None)[0]


NM_P = {m: noise_model(pP[m], rP[m], MP) for m in ("gas", "eui")}
print("noise model from P (a0, b0)", {m: np.round(v, 5).tolist() for m, v in NM_P.items()})


def upper(m, a):
    return np.full(len(a), B["gmax"]) if m == "gas" else B["gmax"] / a


def lower(m, a):
    return {"gas": B["euimin"] * a, "eui": np.full(len(a), B["euimin"]), "elec": B["eamin"] * a}[m]


def upper4(m, a):
    return {"gas": np.full(len(a), B["gmax"]), "eui": B["gmax"] / a, "elec": B["eamax"] * a}[m]


def tail_shift(p, s, T):
    """mean shift of N(p, s^2) truncated above at T: -s * phi(b) / Phi(b), b = (T - p) / s (log-space, stable)"""
    b = (T - p) / s
    return -s * np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))


def apply_S1(pred, d):
    out = {m: pred[m].copy() for m in MET}
    for m in MET:
        for (lo, hi), dd in zip(GROUPS, d[m]):
            g = (MH >= lo) & (MH <= hi)
            out[m][g] *= 1 + dd
    return out


def s3_feature(m, p, nm):
    s = p * np.sqrt(np.clip(nm[0] + nm[1] / MH, 1e-8, None))
    return tail_shift(p, s, upper(m, AH))


def run_chain(splits, d=DP, nm=NM_P, s1=True, s3=True, s4=True):
    """returns dict step -> metric -> full-H prediction; k of S3 cross-fitted over `splits` (list of (fit, app))"""
    st = {"S0": {m: base[m].copy() for m in MET}}
    cur = apply_S1(st["S0"], d) if s1 else {m: st["S0"][m].copy() for m in MET}
    st["S1"] = {m: cur[m].copy() for m in MET}
    ks = {}
    if s3:
        for m in ("gas", "eui"):
            x = s3_feature(m, cur[m], nm[m])
            e = y[m] - cur[m]
            new = cur[m].copy()
            ks[m] = []
            for fit, app in splits:
                k = float(np.dot(x[fit], e[fit]) / np.dot(x[fit], x[fit]))
                ks[m].append(k)
                new[app] = cur[m][app] + k * x[app]
            cur[m] = new
    st["S3"] = {m: cur[m].copy() for m in MET}
    if s4:
        for m in MET:
            cur[m] = np.clip(cur[m], lower(m, AH), upper4(m, AH))
    st["S4"] = {m: cur[m].copy() for m in MET}
    return st, ks


def total_change(st, frm="S0", to="S4", idx=None):
    per = {m: nrmse(m, st[to][m], idx) - nrmse(m, st[frm][m], idx) for m in MET}
    return sum(per.values()) / 5, per


def kf(n_splits, seed):
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(nH)))


# ---- (a) the claim's protocol -----------------------------------------------------------------------------------
st0, ks0 = run_chain(kf(5, 0))
tot, per = total_change(st0)
steps = {}
for a, b in (("S0", "S1"), ("S1", "S3"), ("S3", "S4")):
    t, p_ = total_change(st0, a, b)
    steps[f"{a}->{b}"] = dict(total=t, **p_)
print(f"(a) claim protocol 5-fold seed0: total {tot:+.8f} per metric " + " ".join(f"{m} {v:+.7f}" for m, v in per.items()))
for k, v in steps.items():
    print(f"     step {k}: total {v['total']:+.8f} | " + " ".join(f"{m} {v[m]:+.7f}" for m in MET))
print("     k per fold", {m: np.round(v, 4).tolist() for m, v in ks0.items()})
res["a_claim_protocol"] = dict(total=tot, per=per, steps=steps, k=ks0)

# ---- (b) other splits -------------------------------------------------------------------------------------------
for nsp, seeds in ((5, range(1, 101)), (2, range(1, 101)), (10, range(1, 21))):
    tots, s3s = [], []
    for sd in seeds:
        st, _ = run_chain(kf(nsp, sd))
        tots.append(total_change(st)[0])
        s3s.append(total_change(st, "S1", "S3")[0])
    tots, s3s = np.array(tots), np.array(s3s)
    print(f"(b) {nsp:2d}-fold seeds {seeds.start}..{seeds.stop - 1}: total mean {tots.mean():+.8f} sd {tots.std():.8f} "
          f"min {tots.min():+.8f} max {tots.max():+.8f} | S3 step mean {s3s.mean():+.8f} sd {s3s.std():.8f}")
    res[f"b_{nsp}fold"] = dict(mean=float(tots.mean()), sd=float(tots.std()), min=float(tots.min()), max=float(tots.max()),
                               s3_mean=float(s3s.mean()), s3_sd=float(s3s.std()))

# split-half with SCORING on the held-out half only (fit k on one half, score the other; both directions)
tots = []
for sd in range(1, 51):
    perm = np.random.default_rng(10_000 + sd).permutation(nH)
    h1, h2 = np.sort(perm[: nH // 2]), np.sort(perm[nH // 2:])
    for fit, app in ((h1, h2), (h2, h1)):
        st, _ = run_chain([(fit, app)])
        tots.append(total_change(st, idx=app)[0])
tots = np.array(tots)
print(f"(b') split-half scored on held-out half only (100 halves): mean {tots.mean():+.8f} sd {tots.std():.8f} "
      f"frac<0 {np.mean(tots < 0):.2f}")
res["b_splithalf_heldout_only"] = dict(mean=float(tots.mean()), sd=float(tots.std()), frac_neg=float(np.mean(tots < 0)))

# ---- (c) uncertainty and concentration --------------------------------------------------------------------------
# delta-method SE: NRMSE change ~ mean(e1^2 - e0^2) / (2 * rmse0 * sigma)
se_parts, contrib = {}, {}
dtot = np.zeros(nH)
for m in MET:
    e0 = (y[m] - st0["S0"][m]) ** 2
    e1 = (y[m] - st0["S4"][m]) ** 2
    rm0 = np.sqrt(e0.mean())
    di = (e1 - e0) / (2 * rm0 * SIG[m]) / nH  # per-row contribution to the NRMSE change (first order)
    contrib[m] = di
    dtot += di / 5
    se_parts[m] = float(np.std(e1 - e0) / np.sqrt(nH) / (2 * rm0 * SIG[m]))
se_tot = float(np.std(dtot) * np.sqrt(nH))
print(f"(c) first-order total {dtot.sum():+.8f}, delta-method SE of total {se_tot:.8f} (z {dtot.sum() / se_tot:+.2f}); "
      f"per-metric SE " + " ".join(f"{m} {v:.7f}" for m, v in se_parts.items()))
rng = np.random.default_rng(123)
bt = np.array([dtot[rng.integers(0, nH, nH)].sum() for _ in range(2000)])
print(f"    bootstrap (2000) of first-order total: mean {bt.mean():+.8f} sd {bt.std():.8f} frac<0 {np.mean(bt < 0):.3f} "
      f"q05 {np.quantile(bt, 0.05):+.8f} q95 {np.quantile(bt, 0.95):+.8f}")
order = np.argsort(dtot)
conc = {}
for kk in (1, 3, 5, 10, 20, 50):
    conc[kk] = float(dtot[order[kk:]].sum())
print("    total without the k most-improved rows:", {k: f"{v:+.8f}" for k, v in conc.items()})
changed = np.zeros(nH, bool)
for m in MET:
    changed |= st0["S4"][m] != st0["S0"][m]
print(f"    rows changed in any metric {changed.sum()}, rows improved {(dtot < 0).sum()}, worsened {(dtot > 0).sum()}")
# by segment
seg = {"M2-3": (MH >= 2) & (MH <= 3), "M4": MH == 4, "other": MH >= 5}
segres = {}
for sname, g in seg.items():
    segres[sname] = {m: float(contrib[m][g].sum()) for m in MET}
    segres[sname]["total"] = float(dtot[g].sum())
print("    first-order contribution by segment", {s: {k: f"{v:+.8f}" for k, v in d.items()} for s, d in segres.items()})
# S1-only EUI concentration
s1e = {}
for m in MET:
    e0 = (y[m] - st0["S0"][m]) ** 2
    e1 = (y[m] - st0["S1"][m]) ** 2
    di = (e1 - e0) / (2 * np.sqrt(e0.mean()) * SIG[m]) / nH
    o = np.argsort(di)
    s1e[m] = {"all": float(di.sum()), "wo_top1": float(di[o[1:]].sum()), "wo_top3": float(di[o[3:]].sum()),
              "wo_top5": float(di[o[5:]].sum()), "wo_worst5": float(di[o[:-5]].sum()),
              "n_changed": int((st0["S1"][m] != st0["S0"][m]).sum()),
              "n_improved": int((di < 0).sum()), "n_worsened": int((di > 0).sum())}
print("    S1-only per metric (first order, NRMSE units; divide by 5 for total):",
      {m: {k: (f"{v:+.7f}" if isinstance(v, float) else v) for k, v in d.items()} for m, d in s1e.items()})
res["c_uncertainty"] = dict(first_order_total=float(dtot.sum()), se_delta=se_tot, se_per_metric=se_parts,
                            boot_mean=float(bt.mean()), boot_sd=float(bt.std()), boot_frac_neg=float(np.mean(bt < 0)),
                            wo_top=conc, by_segment=segres, s1_only=s1e, rows_changed=int(changed.sum()))

# ---- (d) reverse direction for S1: fit deltas on H (corrected residuals), apply to the P teacher, score on P ----------
rH = {m: y[m] - base[m] for m in MET}
DH = deltas_from(base, rH, MH)
print("(d) S1 deltas fitted on H (corrected):", {m: [round(v, 4) for v in d] for m, d in DH.items()})
rev = {}
for m in MET:
    p = pP[m].copy()
    for (lo, hi), dd in zip(GROUPS, DH[m]):
        g = (MP >= lo) & (MP <= hi)
        p[g] *= 1 + dd
    rev[m] = (np.sqrt(np.mean((yP[m] - p) ** 2)) - np.sqrt(np.mean(rP[m] ** 2))) / SIG[m]
rev_tot = sum(rev.values()) / 5
print(f"    H->P S1 change on P rows: total {rev_tot:+.8f} | " + " ".join(f"{m} {v:+.7f}" for m, v in rev.items()))
# same but S1 deltas from P applied to P teacher would be in-sample, skip; and P->H S1-only (honest, from (a))
res["d_reverse_S1"] = dict(deltas_H=DH, total=float(rev_tot), per=rev)

# ---- (e) variants -------------------------------------------------------------------------------------------------
var = {}
for est in ("ros", "meanrel"):
    d = deltas_from(pP, rP, MP, est)
    st, _ = run_chain(kf(5, 0), d=d)
    var[f"S1 est={est}"] = total_change(st)[0]
st, _ = run_chain(kf(5, 0), s3=False, s4=False)
var["S1 only"] = total_change(st, "S0", "S4")[0]
st, _ = run_chain(kf(5, 0), s1=False)
var["S3+S4 only (5f seed0)"] = total_change(st)[0]
st, _ = run_chain(kf(5, 0), s1=False, s3=False)
var["S4 only"] = total_change(st)[0]
d23 = {m: [DP[m][0], 0.0] for m in MET}
st, _ = run_chain(kf(5, 0), d=d23)
var["S1(M2-3 only)+S3+S4"] = total_change(st)[0]
# S1 on gas+elec only (no EUI) to see how much rests on EUI
dnoe = {m: (DP[m] if m != "eui" else [0.0, 0.0]) for m in MET}
st, _ = run_chain(kf(5, 0), d=dnoe)
var["S1 without EUI + S3+S4"] = total_change(st)[0]
# S3 with noise model fitted on H fit folds instead of P (per fold)
tt = []
for fit, app in kf(5, 7):
    nmH = {m: noise_model(base[m][fit], (y[m] - base[m])[fit], MH[fit]) for m in ("gas", "eui")}
    st, _ = run_chain([(fit, app)], nm=nmH)
    tt.append(total_change(st, idx=app)[0])
var["chain, S3 noise model from H fit folds (5f seed7, mean of fold totals)"] = float(np.mean(tt))
for k, v in var.items():
    print(f"(e) {k:70s} total {v:+.8f}")
res["e_variants"] = var

json.dump(res, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
