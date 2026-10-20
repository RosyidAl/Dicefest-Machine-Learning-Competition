"""Auditor 'tried': cheap post-processing checks on the honest holdout chain (exp/audit/chain_H.npz).
All fits are on P (bounds, noise model) or cross-fitted on H (5-fold, same KFold(5, shuffle, 0) as audit_base.py).
Effects are reported per metric (NRMSE on H) and as change of the TOTAL score (= per-metric change / 5).
Bootstrap sd of each delta (200 resamples of H rows)."""
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(H), len(H)) for _ in range(200)]
KF = list(KFold(5, shuffle=True, random_state=0).split(np.zeros(len(H))))


def nr(y, p, m):
    return np.sqrt(np.mean((y - p) ** 2)) / SIG[m]


def delta(y, p0, p1, m):
    d = nr(y, p1, m) - nr(y, p0, m)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    bs = [np.sqrt(e1[b].mean()) / SIG[m] - np.sqrt(e0[b].mean()) / SIG[m] for b in BOOT]
    return d, float(np.std(bs))


def report(tag, m, y, p0, p1):
    d, sd = delta(y, p0, p1, m)
    print(f"{m:4s} {tag:55s} {nr(y, p1, m):.6f}  delta {d:+.7f} (sd {sd:.7f})  total {d / 5:+.8f}", flush=True)
    return d


def cross_fit(F, r, fitfun):
    pc = np.zeros(len(r))
    for a, b in KF:
        pc[b] = F[b] @ fitfun(F[a], r[a])
    return pc


ols = lambda F, r: np.linalg.lstsq(F, r, rcond=None)[0]  # noqa: E731

res = {}
for m in ("gas", "eui", "elec"):
    y, t, c, F = z[f"{m}_y"], z[f"{m}_teacher"], z[f"{m}_corrected"], z[f"{m}_F"].astype(np.float64)
    rH = y - t
    yP = z[f"{m}_rP"] + z[f"{m}_pP"]  # scale*y on P = y on the metric's scale (gas: A*eui = gas)
    print(f"== {m}: teacher {nr(y, t, m):.6f} corrected(stored) {nr(y, c, m):.6f}", flush=True)
    c64 = t + cross_fit(F, rH, ols)
    print(f"   corrected recomputed from float32 F: {nr(y, c64, m):.6f}", flush=True)
    res[m] = dict(y=y, t=t, c=c, yP=yP)
    # 1. hard bounds from P
    lo, hi = yP.min(), yP.max()
    print(f"   P bounds [{lo:.4g}, {hi:.6g}]; H rows corrected > hi: {(c > hi).sum()}, < lo: {(c < lo).sum()}; "
          f"y_H max {y.max():.6g}", flush=True)
    report("clip to [min_P, max_P]", m, y, c, np.clip(c, lo, hi))
    if m == "eui":
        gmax = res["gas"]["yP"].max()
        bound = gmax / AH
        print(f"   H rows eui corrected > gasmax_P/A: {(c > bound).sum()}", flush=True)
        report("clip eui at gasmax_P / A", m, y, c, np.minimum(c, bound))
        report("clip eui at gasmax_P / A and [min_P,max_P]", m, y, c, np.clip(np.minimum(c, bound), lo, None))
    # 2. linear recalibration y ~ a + b*c (cross-fitted)
    X = np.column_stack([np.ones(len(c)), c])
    pc = np.zeros(len(c))
    for a, b in KF:
        pc[b] = X[b] @ ols(X[a], y[a])
    report("linear recalibration a + b*corrected (5-fold on H)", m, y, c, pc)
    # 3. correction scale factor (cross-fitted)
    corr = c64 - t
    pc = np.zeros(len(c))
    for a, b in KF:
        s = (corr[a] @ (y[a] - t[a])) / (corr[a] @ corr[a])
        pc[b] = t[b] + s * corr[b]
    report("scale on cross-fitted correction (5-fold)", m, y, c64, pc)
    # 4. ridge lambda (cross-fitted), penalty relative to mean diag of F'F
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1):
        def ridge(Fa, ra, alpha=alpha):
            G = Fa.T @ Fa
            return np.linalg.solve(G + alpha * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), Fa.T @ ra)
        report(f"ridge lambda alpha={alpha} (5-fold)", m, y, c64, t + cross_fit(F, rH, ridge))
    # 5. 2-fold and 10-fold cross-fit of OLS (lambda estimation noise)
    for k in (2, 10, 20):
        pc = np.zeros(len(c))
        for a, b in KFold(k, shuffle=True, random_state=0).split(F):
            pc[b] = F[b] @ ols(F[a], rH[a])
        print(f"   OLS lambda {k:2d}-fold cross-fit: {nr(y, t + pc, m):.6f}", flush=True)
    # 6. intercept in the correction
    Fi = np.column_stack([F, np.ones(len(F))])
    report("correction with intercept (5-fold)", m, y, c64, t + cross_fit(Fi, rH, ols))

# 7. soft truncation for gas: E[y | y < c] under y ~ N(mu, s^2), s^2 = mu^2 (a + b/M), (a, b) fitted on P residuals
m = "gas"
y, c, yP = res[m]["y"], res[m]["c"], res[m]["yP"]
pP = z["gas_pP"]
rel2 = (z["gas_rP"] / pP) ** 2
X = np.column_stack([np.ones(len(pP)), 1 / MP])
ab = np.linalg.lstsq(X, rel2, rcond=None)[0]
print("gas noise model on P: rel var = a + b/M:", ab, flush=True)
hi = yP.max()
for bins in ((7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e7)):
    sel = (c >= bins[0]) & (c < bins[1])
    if sel.sum():
        print(f"   H corrected in [{bins[0]:.3g},{bins[1]:.3g}): n={sel.sum()}, mean resid y-c {np.mean(y[sel] - c[sel]):+.0f}"
              f" (se {np.std(y[sel] - c[sel]) / np.sqrt(sel.sum()):.0f})", flush=True)
    selP = (pP >= bins[0]) & (pP < bins[1])
    if selP.sum():
        print(f"   P fitted  in [{bins[0]:.3g},{bins[1]:.3g}): n={selP.sum()}, mean resid {np.mean(z['gas_rP'][selP]):+.0f}",
              flush=True)
for infl in (0.5, 1.0, 1.5):
    s = np.abs(c) * np.sqrt(np.maximum(ab[0] + ab[1] / MH, 1e-6)) * infl
    al = (hi - c) / s
    adj = c - s * norm.pdf(al) / np.maximum(norm.cdf(al), 1e-12)
    adj = np.minimum(adj, hi)
    report(f"soft truncation E[y|y<max_P] (s x {infl})", m, y, c, adj)
# 8. eui soft truncation via the gas bound: eui < gasmax/A
m = "eui"
y, c = res[m]["y"], res[m]["c"]
pPe = z["eui_pP"]
rel2 = (z["eui_rP"] / pPe) ** 2
abe = np.linalg.lstsq(np.column_stack([np.ones(len(pPe)), 1 / MP]), rel2, rcond=None)[0]
bound = res["gas"]["yP"].max() / AH
for infl in (0.5, 1.0, 1.5):
    s = np.abs(c) * np.sqrt(np.maximum(abe[0] + abe[1] / MH, 1e-6)) * infl
    al = (bound - c) / s
    adj = np.minimum(c - s * norm.pdf(al) / np.maximum(norm.cdf(al), 1e-12), bound)
    report(f"eui soft truncation at gasmax_P/A (s x {infl})", m, y, c, adj)
