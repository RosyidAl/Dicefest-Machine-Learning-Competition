"""Verifier (leakage/optimism lens) 1: independent re-derivation of the truncation-shrink claim from chain_H.npz only.

Claim: gas/EUI shrink toward E[y | y < 1e6] with s = k*p*sqrt(a0 + b0/M); a0, b0, k, c fitted on P (teacher P-OOF),
scored on the honest corrected chain prediction on H: total -2.556e-5.

Checks (all parameters fitted on P only unless stated):
 1. re-derive a0, b0, k grid, c, H delta (own code)
 2. family selection on P (normal vs lognormal, with a k grid for both) -> is the chosen family P-selected?
 3. P-parameter uncertainty: bootstrap P rows, refit (a0, b0, k, c) on each, score on fixed H
 4. H bootstrap + split-halves with permuted halves (not the even/odd split)
 5. influence: share of the H gain from the top-n contributing rows; delta after dropping them
 6. teacher vs corrected bias near the bound on H (does the correction already remove part of it?)
 7. same shrink applied to the H teacher (not corrected) as a consistency check
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/v1_rederive.py
"""
import json

import numpy as np
from scipy.stats import norm

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
KGRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
OUT = "exp/audit/verify_integrity/v1_rederive.json"

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
D = {m: dict(pP=z[f"{m}_pP"], rP=z[f"{m}_rP"], y=z[f"{m}_y"], tH=z[f"{m}_teacher"],
             cH=np.clip(z[f"{m}_corrected"], 0, None)) for m in ("gas", "eui")}
out = {}


def var_fit(pP, rP, MP):
    ok = pP > 1000
    X = np.column_stack([np.ones(ok.sum()), 1.0 / MP[ok]])
    return np.linalg.lstsq(X, (rP[ok] / pP[ok]) ** 2, rcond=None)[0]


def adj_normal(pg, m, k, ab):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(ab[0] + ab[1] / m, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def adj_lognormal(pg, m, k, ab):
    v = np.maximum(ab[0] + ab[1] / m, 1e-6) * k * k
    sv = np.sqrt(v)
    lmu = np.log(np.maximum(pg, 1.0)) - v / 2
    ratio = np.exp(norm.logcdf((np.log(T) - lmu - v) / sv) - norm.logcdf((np.log(T) - lmu) / sv))
    return pg * ratio - pg


FAM = dict(normal=adj_normal, lognormal=adj_lognormal)


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def scale(m, part):
    return (AP if part == "P" else AH) if m == "eui" else 1.0


def fit_P(m, fam, ab, idx=None):
    """k (grid) and c on P by least squares on the metric's own scale; returns (k, c, dSSE)"""
    pP, rP, MPx, sc = D[m]["pP"], D[m]["rP"], MP, scale(m, "P")
    if idx is not None:
        pP, rP, MPx = pP[idx], rP[idx], MPx[idx]
        sc = sc[idx] if m == "eui" else 1.0
    best = None
    grid = {}
    for k in KGRID:
        aP = FAM[fam](pP * sc, MPx, k, ab) / sc
        saa = float(np.sum(aP * aP))
        c = float(np.sum(aP * rP) / saa) if saa > 0 else 0.0
        d = -c * c * saa
        grid[k] = (c, d)
        if best is None or d < best[2]:
            best = (k, c, d)
    return best, grid


def apply_H(m, fam, ab, k, c, which="cH"):
    p = D[m][which]
    sc = scale(m, "H")
    return p + c * FAM[fam](p * sc, MH, k, ab) / sc


# 1. re-derivation
ab_gas = var_fit(D["gas"]["pP"], D["gas"]["rP"], MP)
out["var_model_P"] = ab_gas.tolist()
res = {}
tot = 0.0
for m in ("gas", "eui"):
    (k, c, d), grid = fit_P(m, "normal", ab_gas)
    pa = apply_H(m, "normal", ab_gas, k, c)
    dt = (nr(D[m]["y"], pa, m) - nr(D[m]["y"], D[m]["cH"], m)) / 5
    res[m] = dict(k=k, c=c, base=nr(D[m]["y"], D[m]["cH"], m), new=nr(D[m]["y"], pa, m), delta_total=dt,
                  grid={str(kk): dict(c=v[0], dSSE_P=v[1],
                                      H_delta_total=(nr(D[m]["y"], apply_H(m, "normal", ab_gas, kk, v[0]), m)
                                                     - nr(D[m]["y"], D[m]["cH"], m)) / 5) for kk, v in grid.items()})
    tot += dt
res["total"] = tot
out["rederive_normal"] = res

# 2. family selection on P (both families, k grid each; lognormal variance from the log residuals on P)
pP, rP = D["gas"]["pP"], D["gas"]["rP"]
lr = np.log(np.maximum(pP + rP, 1.0)) - np.log(np.maximum(pP, 1.0))
ab_log = np.linalg.lstsq(np.column_stack([np.ones(len(lr)), 1.0 / MP]), (lr - lr.mean()) ** 2, rcond=None)[0]
fam_sel = {}
for fam, ab in (("normal", ab_gas), ("lognormal", ab_log)):
    r = {}
    for m in ("gas", "eui"):
        (k, c, d), _ = fit_P(m, fam, ab)
        pa = apply_H(m, fam, ab, k, c)
        r[m] = dict(k=k, c=c, dSSE_P=d, H_delta_total=(nr(D[m]["y"], pa, m) - nr(D[m]["y"], D[m]["cH"], m)) / 5)
    r["sum_dSSE_P_rel"] = r["gas"]["dSSE_P"] / float(np.sum(D["gas"]["rP"] ** 2)) + \
        r["eui"]["dSSE_P"] / float(np.sum(D["eui"]["rP"] ** 2))
    r["H_total"] = r["gas"]["H_delta_total"] + r["eui"]["H_delta_total"]
    fam_sel[fam] = r
fam_sel["P_selects"] = min(("normal", "lognormal"), key=lambda f: fam_sel[f]["sum_dSSE_P_rel"])
out["family_selection"] = fam_sel

# 3. P-parameter uncertainty: bootstrap P rows, refit a0, b0, k, c; score on fixed H
rng = np.random.default_rng(123)
pb = []
for b in range(200):
    idx = rng.integers(0, len(P), len(P))
    ab = var_fit(D["gas"]["pP"][idx], D["gas"]["rP"][idx], MP[idx])
    t = 0.0
    for m in ("gas", "eui"):
        (k, c, d), _ = fit_P(m, "normal", ab, idx)
        pa = apply_H(m, "normal", ab, k, c)
        t += (nr(D[m]["y"], pa, m) - nr(D[m]["y"], D[m]["cH"], m)) / 5
    pb.append(t)
pb = np.array(pb)
out["P_boot_refit_H_total"] = dict(mean=float(pb.mean()), sd=float(pb.std()), q05=float(np.quantile(pb, 0.05)),
                                   q95=float(np.quantile(pb, 0.95)), frac_worse=float((pb > 0).mean()))

# 4/5. H bootstrap, permuted halves, influence (final claimed params)
CL = dict(a0=0.008064793880232542, b0=0.3929692979589114, k=0.7, c_gas=1.0126184473794142, c_eui=1.141696475118523)
abc = (CL["a0"], CL["b0"])
pa_cl = {m: apply_H(m, "normal", abc, CL["k"], CL[f"c_{m}"]) for m in ("gas", "eui")}
e0 = {m: (D[m]["y"] - D[m]["cH"]) ** 2 for m in ("gas", "eui")}
e1 = {m: (D[m]["y"] - pa_cl[m]) ** 2 for m in ("gas", "eui")}


def tot_delta(rows):
    return sum((np.sqrt(e1[m][rows].mean()) - np.sqrt(e0[m][rows].mean())) / SIG[m] / 5 for m in ("gas", "eui"))


allr = np.arange(len(H))
out["claimed_params_H_total"] = float(tot_delta(allr))
out["claimed_params_H_per_metric"] = {m: float((np.sqrt(e1[m].mean()) - np.sqrt(e0[m].mean())) / SIG[m] / 5)
                                      for m in ("gas", "eui")}
rng = np.random.default_rng(7)
bs = np.array([tot_delta(rng.integers(0, len(H), len(H))) for _ in range(500)])
out["H_boot_total"] = dict(sd=float(bs.std()), frac_worse=float((bs > 0).mean()))
halves = []
for s in range(20):
    perm = np.random.default_rng(1000 + s).permutation(len(H))
    halves.append([float(tot_delta(perm[: len(H) // 2])), float(tot_delta(perm[len(H) // 2:]))])
halves = np.array(halves)
out["perm_halves"] = dict(n_splits=20, frac_halves_positive=float((halves > 0).mean()),
                          min=float(halves.min()), max=float(halves.max()))
# influence: per-row gain on the total scale (linearized: d sqrt(mean e) ~ (mean e1 - mean e0)/(2 rmse))
contrib = sum((e1[m] - e0[m]) / (2 * np.sqrt(e0[m].mean()) * len(H)) / SIG[m] / 5 for m in ("gas", "eui"))
order = np.argsort(contrib)
infl = {}
for n in (1, 3, 5, 10, 20):
    keep = np.setdiff1d(allr, order[:n])
    infl[f"drop_top{n}_gain_rows"] = float(tot_delta(keep))
infl["rows_changed_gas_gt_1pct"] = int((np.abs(pa_cl["gas"] - D["gas"]["cH"]) > 0.01 * D["gas"]["cH"]).sum())
infl["sum_contrib_negative"] = float(contrib[contrib < 0].sum())
infl["sum_contrib_positive"] = float(contrib[contrib > 0].sum())
infl["n_rows_contrib_nonzero_1e-9"] = int((np.abs(contrib) > 1e-9).sum())
out["influence"] = infl

# 6. teacher vs corrected bias near the bound on H (and residual after shrink)
bands = ((7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 2e6))
bias = {}
for which in ("tH", "cH"):
    p = D["gas"][which]
    y = D["gas"]["y"]
    pa = apply_H("gas", "normal", abc, CL["k"], CL["c_gas"], which)
    bias[which] = {f"{lo:.0f}-{hi:.0f}": dict(n=int(((p >= lo) & (p < hi)).sum()),
                                              resid=float((y - p)[(p >= lo) & (p < hi)].mean()),
                                              resid_after=float((y - pa)[(p >= lo) & (p < hi)].mean()),
                                              se=float((y - p)[(p >= lo) & (p < hi)].std()
                                                       / np.sqrt(max(((p >= lo) & (p < hi)).sum(), 1))))
                   for lo, hi in bands}
out["H_bias_bands_gas"] = bias
# 7. consistency on the H teacher
t = 0.0
for m in ("gas", "eui"):
    pa = apply_H(m, "normal", abc, CL["k"], CL[f"c_{m}"], "tH")
    t += (nr(D[m]["y"], pa, m) - nr(D[m]["y"], D[m]["tH"], m)) / 5
out["teacher_H_total"] = t
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
