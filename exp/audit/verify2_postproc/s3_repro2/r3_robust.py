"""verify2_postproc second pass, S3 robustness (chain_H.npz only):
  - bootstrap of the H gain with k fitted on the disjoint P side (honest, fixed rule),
  - sensitivity to the noise scale s (x0.5 .. x2) and to the cap T, k always fitted on P and scored on H,
  - decomposition: rows predicted above the cap vs below; hard clip alone; leave-top-rows-out,
  - expected test gain from the H/test ratio of mean squared S3 change.
Totals = per-target NRMSE change / 5. S1 is applied first (P-side deltas, ratio of sums), as in the claim."""
import json

import numpy as np
from scipy.stats import norm

OUT = "exp/audit/verify2_postproc/s3_repro2"
SIG = {"gas": 151750.04, "eui": 40.65}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AH, MH, AP, MP = A[H], M[H], A[P], M[P]
T_P = float((z["gas_pP"] + z["gas_rP"]).max())
T_ALL = 999764.8988
log = open(f"{OUT}/r3_robust.log", "w")
res = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


def adj_tn(p, Tm, Mr, a0, b0, scale=1.0):
    s = scale * p * np.sqrt(a0 + b0 / Mr)
    b = (Tm - p) / s
    return -s * np.exp(norm.logpdf(b) - norm.logcdf(b))


def mse(y, p):
    return float(np.mean((y - p) ** 2))


sd, pH, yH = {}, {}, {}
for m in ("gas", "eui"):
    rP, pP = z[m + "_rP"], z[m + "_pP"]
    sd[m] = np.linalg.lstsq(np.column_stack([np.ones(len(P)), 1.0 / MP]), (rP / pP) ** 2, rcond=None)[0]
    p = z[m + "_corrected"].copy()
    for gp, gh in ((MP <= 3, MH <= 3), (MP == 4, MH == 4)):
        p[gh] *= 1 + rP[gp].sum() / pP[gp].sum()
    pH[m], yH[m] = p, z[m + "_y"]


def fit_on_P(m, Tc, scale=1.0):
    pP, rP = z[m + "_pP"], z[m + "_rP"]
    a = adj_tn(pP, Tc if m == "gas" else Tc / AP, MP, *sd[m], scale)
    return float(a @ rP / (a @ a))


def gain_H(m, k, Tc, scale=1.0, rows=None):
    y, p = yH[m], pH[m]
    a = adj_tn(p, Tc if m == "gas" else Tc / AH, MH, *sd[m], scale)
    if rows is not None:
        y, p, a = y[rows], p[rows], a[rows]
    base = np.sqrt(mse(y, p))
    return (np.sqrt(mse(y, p + k * a)) - base) / SIG[m]


# 1. bootstrap of the honest P->H gain (rule fixed: k from P), 2000 resamples of H rows
rng = np.random.default_rng(7)
kP = {m: fit_on_P(m, T_P) for m in ("gas", "eui")}
dl = {}
for m in ("gas", "eui"):
    y, p = yH[m], pH[m]
    a = adj_tn(p, T_P if m == "gas" else T_P / AH, MH, *sd[m])
    dl[m] = ((y - p - kP[m] * a) ** 2, (y - p) ** 2)
n = len(H)
boot = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    t = 0.0
    for m in ("gas", "eui"):
        t += (np.sqrt(dl[m][0][i].mean()) - np.sqrt(dl[m][1][i].mean())) / SIG[m] / 5
    boot.append(t)
boot = np.array(boot)
point = sum((np.sqrt(dl[m][0].mean()) - np.sqrt(dl[m][1].mean())) / SIG[m] / 5 for m in ("gas", "eui"))
res["boot"] = dict(point=float(point), mean=float(boot.mean()), sd=float(boot.std()), frac_neg=float((boot < 0).mean()),
                   q05=float(np.quantile(boot, 0.05)), q95=float(np.quantile(boot, 0.95)))
say(f"P->H rule (k gas {kP['gas']:.4f}, eui {kP['eui']:.4f}): total {point:+.8f}; bootstrap over H rows sd {boot.std():.8f} "
    f"frac<0 {(boot < 0).mean():.3f} 90% [{np.quantile(boot, 0.05):+.8f}, {np.quantile(boot, 0.95):+.8f}]")

# 2. sensitivity to s scale and to T (k always from P)
sens = {}
for Tc, tl in ((T_P, "T_P"), (T_ALL, "T_all"), (1e6, "1e6")):
    for sc in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        t = 0.0
        ks = {}
        for m in ("gas", "eui"):
            ks[m] = fit_on_P(m, Tc, sc)
            t += gain_H(m, ks[m], Tc, sc) / 5
        sens[f"{tl}_x{sc}"] = dict(total=float(t), k=ks)
        say(f"sens {tl:5s} s x{sc:<4}: k gas {ks['gas']:.3f} eui {ks['eui']:.3f}  total {t:+.8f}")
res["sens"] = sens

# 3. decomposition on H (k from P, T_P): over-cap rows vs the rest; hard clip only; leave top rows out
dec = {}
for m in ("gas", "eui"):
    y, p = yH[m], pH[m]
    Tm = T_P if m == "gas" else T_P / AH
    a = adj_tn(p, Tm, MH, *sd[m])
    over = p > Tm
    base_sse = np.sum((y - p) ** 2)
    rm = np.sqrt(base_sse / n)
    d_rows = (y - p - kP[m] * a) ** 2 - (y - p) ** 2
    to_total = lambda s: float(s / n / (2 * rm) / SIG[m] / 5)  # noqa: E731  (first-order)
    clip = np.minimum(p, Tm)
    dec[m] = dict(over_cap_rows=int(over.sum()), total_from_over=to_total(d_rows[over].sum()),
                  total_from_rest=to_total(d_rows[~over].sum()),
                  hardclip_total=float((np.sqrt(mse(y, clip)) - rm) / SIG[m] / 5))
    order = np.argsort(d_rows)
    for drop in (0, 3, 5, 10, 20):
        keep = np.ones(n, bool)
        keep[order[:drop]] = False
        dec[m][f"drop_best{drop}"] = float(gain_H(m, kP[m], T_P, 1.0, keep) / 5)
    say(f"decomp {m}: over-cap rows {over.sum()} contribute {dec[m]['total_from_over']:+.8f}, rest {dec[m]['total_from_rest']:+.8f}; "
        f"hard clip only {dec[m]['hardclip_total']:+.8f}; drop best 0/3/5/10/20 rows: "
        + " ".join(f"{dec[m][f'drop_best{d}']:+.8f}" for d in (0, 3, 5, 10, 20)))
    # the same on the P side (independent rows, teacher), k from P is in-sample there: report the bias only
res["decomp"] = dec
json.dump(res, open(f"{OUT}/r3_robust.json", "w"), indent=1)
log.close()
