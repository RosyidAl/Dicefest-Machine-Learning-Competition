"""Independent reproduction (verify2_tried) of the gas upper-bound fix G on the holdout chain (exp/audit/chain_H.npz).

Fix G (claimed): for a gas prediction g > K (K = 9e5): g' = min(g + beta * (g - K), CAP), CAP = 1e6;
EUI e is fixed through its implied gas e*A the same way and divided back by A.
beta = h.r / h.h with h = max(p - K, 0) and r the residual, fitted on rows that are NOT scored.

Honest splits used here:
  S0  beta from the P side (teacher pP / residual rP), applied to H corrected predictions (the auditor's split).
  S1  split-half inside H (200 random halvings): beta fitted on one half from the CORRECTED prediction's residuals,
      applied to the other half, and vice versa (every H row is scored with a beta that never saw it).
  S2  as S1, but beta fitted on the half's TEACHER residuals (mirrors the pipeline action, which fits on teacher pf).
  S3  5-fold cross-fit inside H with seeds 101..120 (other fold seeds than the auditor's).
Effects are reported as TOTAL score change = (d NRMSE gas + d NRMSE eui) / 5 (elec untouched).
"""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65}
K, CAP = 9e5, 1e6
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, P = z["H"], z["P"]
AH = z["A"][H]
yg, ye = z["gas_y"], z["eui_y"]
cg, ce = z["gas_corrected"], z["eui_corrected"]
tg, te_ = z["gas_teacher"], z["eui_teacher"]
nH = len(H)


def nrmse(y, p, s):
    return np.sqrt(np.mean((y - p) ** 2)) / s


def fix_g(g, beta):
    return np.where(g > K, np.minimum(g + beta * (g - K), CAP), g)


def fix_e(e, beta):
    g = e * AH if np.ndim(beta) == 0 else e * AH
    return np.where(g > K, np.minimum(g + beta * (g - K), CAP) / AH, e)


def beta_fit(p, r):
    h = np.maximum(p - K, 0.0)
    return float(h @ r / (h @ h))


def total_change(gpred, epred):
    dg = nrmse(yg, gpred, SIG["gas"]) - nrmse(yg, cg, SIG["gas"])
    de = nrmse(ye, epred, SIG["eui"]) - nrmse(ye, ce, SIG["eui"])
    return dg, de, (dg + de) / 5


print(f"H rows {nH}; base NRMSE corrected gas {nrmse(yg, cg, SIG['gas']):.6f} eui {nrmse(ye, ce, SIG['eui']):.6f}; "
      f"teacher gas {nrmse(yg, tg, SIG['gas']):.6f} eui {nrmse(ye, te_, SIG['eui']):.6f}")
print(f"identity check on H: max |eui_y*A - gas_y|/gas_y = {np.max(np.abs(ye * AH - yg) / yg):.2e}; "
      f"max y gas on H {yg.max():.1f}")
print(f"H rows with corrected gas > 9e5: {np.sum(cg > K)}, > 1e6: {np.sum(cg > CAP)}; implied gas eui*A > 9e5: "
      f"{np.sum(ce * AH > K)}; teacher gas > 9e5: {np.sum(tg > K)}")

# bias table by prediction bin (conditioning on the prediction only, not on y)
print("mean residual y - pred by corrected-gas bin (H):")
for lo, hi in [(6e5, 7e5), (7e5, 8e5), (8e5, 8.5e5), (8.5e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 2e6)]:
    m = (cg >= lo) & (cg < hi)
    if m.sum() == 0:
        continue
    r = yg[m] - cg[m]
    rt = yg[m] - tg[m]
    print(f"  [{lo:9.0f},{hi:9.0f}) n={m.sum():4d} corrected resid {r.mean():+9.0f} (se {r.std(ddof=1) / np.sqrt(m.sum()):6.0f})"
          f"  teacher resid on same rows {rt.mean():+9.0f}   max y {yg[m].max():9.0f}")
m = ce * AH >= 9.5e5
print(f"  EUI implied-gas >= 9.5e5: n={m.sum()} mean resid (gas units) {np.mean((ye[m] - ce[m]) * AH[m]):+.0f}")

# S0: P-side teacher fit -> H corrected
b0 = beta_fit(z["gas_pP"], z["gas_rP"])
dg, de, tot = total_change(fix_g(cg, b0), fix_e(ce, b0))
print(f"S0 beta from P teacher {b0:+.4f}: d gas {dg:+.7f} d eui {de:+.7f} TOTAL {tot:+.8f}; rows changed gas "
      f"{np.sum(fix_g(cg, b0) != cg)}, eui {np.sum(np.abs(fix_e(ce, b0) - ce) > 1e-9 * ce)}")
# bootstrap (rows of H) of S0 total change: per-row squared-error deltas, ratio of means
g1, e1 = fix_g(cg, b0), fix_e(ce, b0)
rng = np.random.default_rng(2024)
sg0, sg1 = (yg - cg) ** 2, (yg - g1) ** 2
se0, se1 = (ye - ce) ** 2, (ye - e1) ** 2
bs = []
for _ in range(2000):
    i = rng.integers(0, nH, nH)
    d = (np.sqrt(sg1[i].mean()) - np.sqrt(sg0[i].mean())) / SIG["gas"] + \
        (np.sqrt(se1[i].mean()) - np.sqrt(se0[i].mean())) / SIG["eui"]
    bs.append(d / 5)
bs = np.array(bs)
print(f"S0 bootstrap over H rows (n={nH}, ~private size): mean {bs.mean():+.8f} sd {bs.std():.8f} "
      f"P(<0) {np.mean(bs < 0):.3f}  5%-95% [{np.quantile(bs, .05):+.8f}, {np.quantile(bs, .95):+.8f}]")
# public-size subsets (30% of the test ~ 55,196 rows): resample without replacement 55k rows of H
pub = []
for _ in range(2000):
    i = rng.choice(nH, 55196, replace=False)
    d = (np.sqrt(sg1[i].mean()) - np.sqrt(sg0[i].mean())) / SIG["gas"] + \
        (np.sqrt(se1[i].mean()) - np.sqrt(se0[i].mean())) / SIG["eui"]
    pub.append(d / 5)
pub = np.array(pub)
print(f"S0 public-size subsets: mean {pub.mean():+.8f} sd {pub.std():.8f} P(<0) {np.mean(pub < 0):.3f}")

# beta sensitivity on H (reference only; the H-optimal value is in-sample)
print("beta grid on H (in-sample reference):")
for b in [-0.3, -0.5, -0.6, -0.693, -0.7275, -0.8, -0.9, -1.0]:
    dg, de, tot = total_change(fix_g(cg, b), fix_e(ce, b))
    print(f"  beta {b:+.4f}: d gas {dg:+.7f} d eui {de:+.7f} TOTAL {tot:+.8f}")
print(f"  H in-sample optimal beta (corrected residuals): {beta_fit(cg, yg - cg):+.4f}")


# S1/S2: split-half inside H
def split_half(seed, fit_on):
    perm = np.random.default_rng(seed).permutation(nH)
    halves = [perm[: nH // 2], perm[nH // 2:]]
    g2, e2 = cg.copy(), ce.copy()
    betas = []
    for a, b in [(0, 1), (1, 0)]:
        ia, ib = halves[a], halves[b]
        if fit_on == "corrected":
            beta = beta_fit(cg[ia], yg[ia] - cg[ia])
        else:
            beta = beta_fit(tg[ia], yg[ia] - tg[ia])
        betas.append(beta)
        g2[ib] = fix_g(cg[ib], beta)
        ge = ce[ib] * AH[ib]
        e2[ib] = np.where(ge > K, np.minimum(ge + beta * (ge - K), CAP) / AH[ib], ce[ib])
    return total_change(g2, e2), betas


for fit_on, tag in [("corrected", "S1"), ("teacher", "S2")]:
    res = [split_half(1000 + s, fit_on) for s in range(200)]
    tots = np.array([r[0][2] for r in res])
    dgs = np.array([r[0][0] for r in res])
    des = np.array([r[0][1] for r in res])
    bet = np.array([r[1] for r in res]).ravel()
    print(f"{tag} split-half in H, beta fitted on {fit_on} residuals, 200 halvings: TOTAL mean {tots.mean():+.8f} "
          f"sd {tots.std():.8f} P(<0) {np.mean(tots < 0):.3f} min {tots.min():+.8f} max {tots.max():+.8f}; "
          f"d gas {dgs.mean():+.7f} d eui {des.mean():+.7f}; beta median {np.median(bet):+.3f} "
          f"[{np.quantile(bet, .05):+.3f}, {np.quantile(bet, .95):+.3f}]")

# S3: 5-fold cross-fit inside H, other seeds
tots = []
for seed in range(101, 121):
    fold = np.random.default_rng(seed).permutation(nH) % 5
    g2, e2 = cg.copy(), ce.copy()
    for f in range(5):
        a, b = fold != f, fold == f
        beta = beta_fit(cg[a], yg[a] - cg[a])
        g2[b] = fix_g(cg[b], beta)
        ge = ce[b] * AH[b]
        e2[b] = np.where(ge > K, np.minimum(ge + beta * (ge - K), CAP) / AH[b], ce[b])
    tots.append(total_change(g2, e2)[2])
tots = np.array(tots)
print(f"S3 5-fold cross-fit in H (seeds 101-120): TOTAL mean {tots.mean():+.8f} sd over seeds {tots.std():.8f} "
      f"min {tots.min():+.8f} max {tots.max():+.8f}")

# hard cap only and knot variants with P-side fit (robustness of the knot choice)
dg, de, tot = total_change(np.minimum(cg, CAP), np.where(ce * AH > CAP, CAP / AH, ce))
print(f"hard cap only at 1e6: d gas {dg:+.7f} d eui {de:+.7f} TOTAL {tot:+.8f}")
for k in [8e5, 8.5e5, 9e5, 9.5e5]:
    h = np.maximum(z["gas_pP"] - k, 0)
    b = float(h @ z["gas_rP"] / (h @ h))
    g2 = np.where(cg > k, np.minimum(cg + b * (cg - k), CAP), cg)
    ge = ce * AH
    e2 = np.where(ge > k, np.minimum(ge + b * (ge - k), CAP) / AH, ce)
    dg, de, tot = total_change(g2, e2)
    print(f"knot {k:.0f} (beta from P {b:+.3f}): d gas {dg:+.7f} d eui {de:+.7f} TOTAL {tot:+.8f}")
