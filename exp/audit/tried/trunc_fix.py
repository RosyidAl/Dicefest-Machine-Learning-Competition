"""Gas is hard-truncated at 1e6 kWh in the data (no train row >= 1e6, flat density up to the cap), so for every test
row y < 1e6 and E[y|x] = E[y*|x, y* < 1e6]. The blend members predict rates (per meter / per area) and cannot represent
the cap: on P and H the residual of predictions > 9e5 is strongly negative. Honest checks of truncation-aware fixes:
  (a) fitted on P (teacher-level P predictions pP, y on P), applied to the corrected H predictions;
  (b) cross-fitted 5-fold on H.
EUI = gas / A exactly, so EUI predictions get the same fix on the gas scale: eui' = f(eui * A) / A."""
import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
CAP = 1e6
yP = z["gas_rP"] + z["gas_pP"]
pP = z["gas_pP"]
yH, cH = z["gas_y"], z["gas_corrected"]
yHe, cHe = z["eui_y"], z["eui_corrected"]
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(H), len(H)) for _ in range(300)]
KF = list(KFold(5, shuffle=True, random_state=0).split(np.zeros(len(H))))
print("cap check: max y_P", yP.max(), "max y_H", yH.max(), flush=True)


def stats(y, p0, p1, m):
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    d = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m]
    bs = np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] for b in BOOT])
    return d, bs.std(), (bs < 0).mean()


def show(tag, fg, fe=None):
    """fg: adjusted gas on H; fe: adjusted eui on H (default: same multiplicative fix applied via eui*A)"""
    dg, sg, pg = stats(yH, cH, fg, "gas")
    if fe is None:
        fe = cHe
    de, se, pe = stats(yHe, cHe, fe, "eui")
    print(f"{tag:58s} gas {dg:+.7f} (sd {sg:.7f}, P<0 {pg:.2f}) eui {de:+.7f} (sd {se:.7f}) "
          f"TOTAL {(dg + de) / 5:+.8f}  rows changed {(np.abs(fg - cH) > 1e-6).sum()}", flush=True)


def hinge_X(c, knots):
    return np.column_stack([np.maximum(c - k, 0) for k in knots])


def eui_from(fun, c_eui):
    """apply a gas-scale fix to EUI via the implied gas eui*A"""
    g = c_eui * AH
    return fun(g) / AH


# 0. hard cap
show("hard cap gas<=1e6, eui<=1e6/A", np.minimum(cH, CAP), np.minimum(cHe, CAP / AH))
show("hard cap at max y_P", np.minimum(cH, yP.max()), np.minimum(cHe, yP.max() / AH))

# (a) hinge shrink fitted on P (teacher level), applied to H corrected
for knots in ((9e5,), (8.5e5,), (9e5, 9.5e5), (8.5e5, 9.25e5), (8e5, 9e5, 9.5e5)):
    XP = hinge_X(pP, knots)
    beta = np.linalg.lstsq(XP, yP - pP, rcond=None)[0]
    f = lambda c, beta=beta, knots=knots: np.minimum(c + hinge_X(c, knots) @ beta, CAP)  # noqa: E731
    show(f"(a) hinge {knots} fit on P beta={np.round(beta, 3)}", f(cH), eui_from(f, cHe))
# (b) hinge cross-fitted on H
for knots in ((9e5,), (8.5e5,), (9e5, 9.5e5), (8.5e5, 9.25e5)):
    fg = np.zeros(len(H))
    fe = np.zeros(len(H))
    for a, b in KF:
        beta = np.linalg.lstsq(hinge_X(cH[a], knots), yH[a] - cH[a], rcond=None)[0]
        fg[b] = np.minimum(cH[b] + hinge_X(cH[b], knots) @ beta, CAP)
        gb = cHe[b] * AH[b]
        fe[b] = np.minimum(gb + hinge_X(gb, knots) @ beta, CAP) / AH[b]
    show(f"(b) hinge {knots} cross-fitted on H", fg, fe)
# (c) truncated-normal conditional mean, sd = gamma * c * sqrt(a + b/M); (a, b) and gamma chosen on P
rel2 = (z["gas_rP"] / pP) ** 2
ab = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), rel2, rcond=None)[0]


def tn(c, Mv, gamma):
    s = np.abs(c) * np.sqrt(np.maximum(ab[0] + ab[1] / Mv, 1e-6)) * gamma
    al = (CAP - c) / s
    return np.minimum(c - s * norm.pdf(al) / np.maximum(norm.cdf(al), 1e-300), CAP)


grid = np.round(np.arange(0.2, 1.21, 0.05), 2)
lossP = [np.mean((yP - tn(pP, MP, g)) ** 2) for g in grid]
gP = grid[int(np.argmin(lossP))]
print("gamma chosen on P:", gP, "P mse rel to none:", np.round(np.array(lossP) / np.mean((yP - pP) ** 2) - 1, 7)[:: 2],
      flush=True)
show(f"(c) truncated-normal mean, gamma={gP} chosen on P", tn(cH, MH, gP), tn(cHe * AH, MH, gP) / AH)
for g in (0.3, 0.4, 0.5, 0.6, 0.7):
    show(f"    truncated-normal gamma={g} (fixed, for shape)", tn(cH, MH, g), tn(cHe * AH, MH, g) / AH)
# (d) isotonic regression of y on prediction, fitted on P for predictions >= 7e5, applied on H
sel = pP >= 7e5
iso = IsotonicRegression(out_of_bounds="clip").fit(pP[sel], yP[sel])


def fiso(c):
    out = c.copy()
    s = c >= 7e5
    out[s] = np.minimum(c[s], iso.predict(c[s]))
    return np.minimum(out, CAP)


show("(d) isotonic (P, preds >= 7e5), only lowers", fiso(cH), eui_from(fiso, cHe))
# per-bin residuals on H after the P-fitted fix (c)
fg = tn(cH, MH, gP)
for lo, hi in ((7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e7)):
    s = (cH >= lo) & (cH < hi)
    print(f"  H bin [{lo:.3g},{hi:.3g}) n={s.sum()}: resid before {np.mean(yH[s] - cH[s]):+.0f}, after "
          f"{np.mean(yH[s] - fg[s]):+.0f}", flush=True)
