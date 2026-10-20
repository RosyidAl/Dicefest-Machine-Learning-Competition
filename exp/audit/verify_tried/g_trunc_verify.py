# Verify claim G (gas truncation soft cap) honestly on chain_H; also incremental over deployed S1+S3+S4.
import numpy as np
from scipy.stats import norm
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"].astype(float)
SIG = dict(gas=151750.04, eui=40.65)
AH, MP, MH = A[H], M[P], M[H]
KN, CAP = 9e5, 1e6
def nr(m, p): return np.sqrt(np.mean((z[f"{m}_y"] - p) ** 2)) / SIG[m]
yH = {m: z[f"{m}_y"] for m in ("gas", "eui")}
pP, rP = z["gas_pP"], z["gas_rP"]; yP = pP + rP
print("max y_P gas", yP.max(), "max y_H gas", yH["gas"].max(), "n y_P>=9.8e5", (yP >= 9.8e5).sum())
h = np.maximum(pP - KN, 0); beta = float(h @ rP / (h @ h)); print("beta_P", beta, "n P rows pP>9e5", (h > 0).sum())
def G(v, b):
    return np.where(v > KN, np.minimum(v + b * (v - KN), CAP), v)
base = {m: np.clip(z[f"{m}_corrected"], 0, None) for m in ("gas", "eui")}
def applyG(base, b):
    g = G(base["gas"], b); e = G(base["eui"] * AH, b) / AH
    return g, e
g1, e1 = applyG(base, beta)
d = {m: nr(m, x) - nr(m, base[m]) for m, x in (("gas", g1), ("eui", e1))}
print("G vs base: gas %.7f eui %.7f total %.7f" % (d["gas"], d["eui"], (d["gas"] + d["eui"]) / 5))
# bin residuals
for lo, hi in ((9e5, 9.5e5), (9.5e5, 2e6)):
    s = (base["gas"] >= lo) & (base["gas"] < hi)
    print("bin", lo, hi, "n", s.sum(), "mean resid before %.0f after %.0f" % ((yH["gas"] - base["gas"])[s].mean(), (yH["gas"] - g1)[s].mean()))
# bootstrap
rng = np.random.default_rng(0); n = len(H); out = []
eg0 = (yH["gas"] - base["gas"]) ** 2; eg1 = (yH["gas"] - g1) ** 2
ee0 = (yH["eui"] - base["eui"]) ** 2; ee1 = (yH["eui"] - e1) ** 2
for _ in range(300):
    i = rng.integers(0, n, n)
    out.append(((np.sqrt(eg1[i].mean()) - np.sqrt(eg0[i].mean())) / SIG["gas"] + (np.sqrt(ee1[i].mean()) - np.sqrt(ee0[i].mean())) / SIG["eui"]) / 5)
out = np.array(out); print("bootstrap total mean %.7f sd %.7f P(<0) %.3f" % (out.mean(), out.std(), (out < 0).mean()))
# --- deployed S1+S3+S4 analog (deltas from P, kS3 cross-fitted 5-fold on H), then G on top
SD = {}
for m in ("gas", "eui"):
    p_, r_ = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(p_)), 1 / MP]), (r_ / p_) ** 2, rcond=None)[0]
GMAX = yP.max()
def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)
fold = np.random.default_rng(1).integers(0, 5, len(H))
dep = {}
for m in ("gas", "eui"):
    p_, r_ = z[f"{m}_pP"], z[f"{m}_rP"]
    p = base[m].copy()
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + float(p_[gP] @ r_[gP] / (p_[gP] @ p_[gP]))
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / MH, 1e-8, None))
    x = tn_hi(p, s, GMAX if m == "gas" else GMAX / AH)
    out3 = p.copy()
    for f in range(5):
        tr_, te_ = fold != f, fold == f
        k = x[tr_] @ (yH[m] - p)[tr_] / (x[tr_] @ x[tr_]); out3[te_] = p[te_] + k * x[te_]
    if m == "gas": out3 = np.minimum(out3, GMAX)
    else: out3 = np.minimum(out3, GMAX / AH)
    dep[m] = (p, out3)
for m in ("gas", "eui"):
    print(m, "S1 only %.7f  S1+S3+S4 %.7f (vs base)" % (nr(m, dep[m][0]) - nr(m, base[m]), nr(m, dep[m][1]) - nr(m, base[m])))
dd = {m: nr(m, dep[m][1]) - nr(m, base[m]) for m in ("gas", "eui")}
print("deployed-analog total vs base %.7f" % ((dd["gas"] + dd["eui"]) / 5))
# G on top of deployed
dg = {"gas": dep["gas"][1], "eui": dep["eui"][1]}
g2, e2 = applyG(dg, beta)
inc = {m: nr(m, x) - nr(m, dg[m]) for m, x in (("gas", g2), ("eui", e2))}
print("G on top of S1+S3+S4: gas %.7f eui %.7f total %.7f" % (inc["gas"], inc["eui"], (inc["gas"] + inc["eui"]) / 5))
# G instead of S3 (S1 + G + S4)
dg1 = {"gas": dep["gas"][0], "eui": dep["eui"][0]}
g3, e3 = applyG(dg1, beta)
g3 = np.minimum(g3, GMAX); e3 = np.minimum(e3, GMAX / AH)
alt = {m: nr(m, x) - nr(m, dep[m][1]) for m, x in (("gas", g3), ("eui", e3))}
print("S1+G+S4 minus S1+S3+S4: gas %.7f eui %.7f total %.7f" % (alt["gas"], alt["eui"], (alt["gas"] + alt["eui"]) / 5))
