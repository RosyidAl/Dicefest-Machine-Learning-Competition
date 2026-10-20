"""Independent re-check of the small-meter (bld_dim_12 = M) under-prediction claim on the holdout chain.

Uses only exp/audit/chain_H.npz. Groups: G23 = M in {2,3}, G4 = M == 4.
Honest evaluations (fit the per-group shift on one set of rows, score on disjoint rows):
  (1) P -> H : shift fitted on P-side predictions/residuals (pP, rP), applied to H corrected predictions.
  (2) H -> P : shift fitted on H corrected residuals, applied to P-side predictions (reverse direction).
  (3) H internal K-fold cross-fitting (K = 2 and 5, many seeds).
  (4) P internal K-fold cross-fitting.
Every effect is reported as NRMSE change per metric and as change of the TOTAL score (sum / 5).
"""
import json
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M = z["P"], z["H"], z["M"]
MH, MP = M[H], M[P]
GROUPS = {"G23": lambda m: (m >= 2) & (m <= 3), "G4": lambda m: m == 4}
out = {}


def nrmse(y, p, s):
    return float(np.sqrt(np.mean((y - p) ** 2)) / s)


def fit_shift(p, y, mask, form):
    """per-group shift estimated from (p, y) on rows `mask`"""
    pp, rr = p[mask], y[mask] - p[mask]
    if mask.sum() == 0:
        return 0.0
    if form == "ls":  # least-squares multiplicative: argmin_d sum (r - d p)^2
        return float(pp @ rr / (pp @ pp))
    if form == "ros":  # ratio of sums
        return float(rr.sum() / pp.sum())
    if form == "add":
        return float(rr.mean())
    raise ValueError(form)


def apply_shift(p, m, deltas, form):
    q = p.copy()
    for g, d in deltas.items():
        sel = GROUPS[g](m)
        q[sel] = p[sel] * (1 + d) if form in ("ls", "ros") else p[sel] + d
    return np.clip(q, 0, None)


# 0. descriptive: bias per M value on H (teacher, corrected) and on P (pP)
desc = {}
for met in SIG:
    y, t, c = z[f"{met}_y"], z[f"{met}_teacher"], z[f"{met}_corrected"]
    pP, rP = z[f"{met}_pP"], z[f"{met}_rP"]
    yP = pP + rP
    rows = []
    for mv in [2, 3, 4, 5, 6, 7, 8]:
        h, p_ = MH == mv, MP == mv
        rel_t = (y[h] - t[h]) / t[h] if h.any() else np.array([np.nan])
        rel_c = (y[h] - c[h]) / c[h] if h.any() else np.array([np.nan])
        rel_p = rP[p_] / pP[p_] if p_.any() else np.array([np.nan])
        rows.append(dict(M=mv, nH=int(h.sum()), nP=int(p_.sum()),
                         H_teacher_ls=fit_shift(t, y, h, "ls") if h.any() else None,
                         H_corr_ls=fit_shift(c, y, h, "ls") if h.any() else None,
                         H_corr_meanrel=float(np.mean(rel_c)), H_corr_se=float(np.std(rel_c) / np.sqrt(max(h.sum(), 1))),
                         P_ls=fit_shift(pP, yP, p_, "ls") if p_.any() else None,
                         P_meanrel=float(np.mean(rel_p)), P_se=float(np.std(rel_p) / np.sqrt(max(p_.sum(), 1)))))
    desc[met] = rows
out["descriptive"] = desc
for met in SIG:
    print(f"== {met}")
    for r in desc[met]:
        print("  M=%d nH %4d nP %5d | H teacher ls %s corr ls %s meanrel %+.4f (se %.4f) | P ls %s meanrel %+.4f (se %.4f)" % (
            r["M"], r["nH"], r["nP"], "%+.4f" % r["H_teacher_ls"] if r["H_teacher_ls"] is not None else "  nan ",
            "%+.4f" % r["H_corr_ls"] if r["H_corr_ls"] is not None else "  nan ", r["H_corr_meanrel"], r["H_corr_se"],
            "%+.4f" % r["P_ls"] if r["P_ls"] is not None else "  nan ", r["P_meanrel"], r["P_se"]))

# 1. P -> H and 2. H -> P
res = {}
for form in ["ls", "ros", "add"]:
    tot_ph, tot_hp = 0.0, 0.0
    for met in SIG:
        y, c = z[f"{met}_y"], z[f"{met}_corrected"]
        pP, rP = z[f"{met}_pP"], z[f"{met}_rP"]
        yP = pP + rP
        dP = {g: fit_shift(pP, yP, GROUPS[g](MP), form) for g in GROUPS}
        dH = {g: fit_shift(c, y, GROUPS[g](MH), form) for g in GROUPS}
        ph = nrmse(y, apply_shift(c, MH, dP, form), SIG[met]) - nrmse(y, c, SIG[met])
        hp = nrmse(yP, apply_shift(pP, MP, dH, form), SIG[met]) - nrmse(yP, pP, SIG[met])
        # only-G23 / only-G4 versions of P -> H
        ph23 = nrmse(y, apply_shift(c, MH, {"G23": dP["G23"]}, form), SIG[met]) - nrmse(y, c, SIG[met])
        ph4 = nrmse(y, apply_shift(c, MH, {"G4": dP["G4"]}, form), SIG[met]) - nrmse(y, c, SIG[met])
        res[f"{form}_{met}"] = dict(dP=dP, dH=dH, P_to_H=ph, H_to_P=hp, P_to_H_G23only=ph23, P_to_H_G4only=ph4)
        tot_ph += ph / 5
        tot_hp += hp / 5
        print(f"[{form:3s}] {met:4s} dP {dP['G23']:+.4f}/{dP['G4']:+.4f} dH {dH['G23']:+.4f}/{dH['G4']:+.4f} | "
              f"P->H dNRMSE {ph:+.7f} (G23 {ph23:+.7f}, G4 {ph4:+.7f}) | H->P dNRMSE {hp:+.7f}")
    res[f"{form}_TOTAL"] = dict(P_to_H=tot_ph, H_to_P=tot_hp)
    print(f"[{form:3s}] TOTAL P->H {tot_ph:+.8f}  H->P {tot_hp:+.8f}")
out["cross"] = res


# 3./4. internal K-fold cross-fitting
def kfold_cf(p, y, m, sig, K, seed, form="ls"):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, K, len(p))  # random fold labels (independent of the row order)
    q = p.copy()
    for k in range(K):
        tr_, te_ = fold != k, fold == k
        for g in GROUPS:
            gm = GROUPS[g](m)
            d = fit_shift(p, y, gm & tr_, form) if (gm & tr_).any() else 0.0
            sel = gm & te_
            q[sel] = p[sel] * (1 + d) if form in ("ls", "ros") else p[sel] + d
    q = np.clip(q, 0, None)
    return nrmse(y, q, sig) - nrmse(y, p, sig)


cf = {}
for name, K in [("H_K2", 2), ("H_K5", 5), ("P_K2", 2), ("P_K5", 5)]:
    tots = []
    per = {met: [] for met in SIG}
    for seed in range(200):
        t = 0.0
        for met in SIG:
            if name.startswith("H"):
                y, p, m = z[f"{met}_y"], z[f"{met}_corrected"], MH
            else:
                p, m = z[f"{met}_pP"], MP
                y = p + z[f"{met}_rP"]
            d = kfold_cf(p, y, m, SIG[met], K, 1000 + seed)
            per[met].append(d)
            t += d / 5
        tots.append(t)
    tots = np.array(tots)
    cf[name] = dict(mean_total=float(tots.mean()), sd_over_seeds=float(tots.std()),
                    frac_negative=float((tots < 0).mean()),
                    per_metric_mean={met: float(np.mean(per[met])) for met in SIG})
    print(f"{name}: total mean {tots.mean():+.8f} sd(seeds) {tots.std():.8f} frac<0 {(tots < 0).mean():.3f} | "
          + " ".join(f"{met} {np.mean(per[met]):+.7f}" for met in SIG))
out["kfold_crossfit"] = cf

# 5. bootstrap sd of the P -> H total gain (resample H rows; P-fitted deltas fixed)
rng = np.random.default_rng(12345)
B = 2000
boot = np.zeros(B)
cache = {}
for met in SIG:
    y, c = z[f"{met}_y"], z[f"{met}_corrected"]
    pP, rP = z[f"{met}_pP"], z[f"{met}_rP"]
    dP = {g: fit_shift(pP, pP + rP, GROUPS[g](MP), "ls") for g in GROUPS}
    cache[met] = (y, c, apply_shift(c, MH, dP, "ls"))
n = len(H)
for b in range(B):
    idx = rng.integers(0, n, n)
    t = 0.0
    for met in SIG:
        y, c, q = cache[met]
        t += (nrmse(y[idx], q[idx], SIG[met]) - nrmse(y[idx], c[idx], SIG[met])) / 5
    boot[b] = t
out["bootstrap_P_to_H_ls"] = dict(mean=float(boot.mean()), sd=float(boot.std()), frac_negative=float((boot < 0).mean()))
print(f"bootstrap P->H ls total: mean {boot.mean():+.8f} sd {boot.std():.8f} frac<0 {(boot < 0).mean():.3f}")
json.dump(out, open("exp/audit/verify2_postproc/v01_chainH.json", "w"), indent=1)
