"""verify2_segments: independent reproduction of the gas 1e6 truncation calibration on the holdout chain.

Own implementation (scipy.special.log_ndtr based inverse Mills ratio), own splits:
  (a) k fitted on the P side (teacher OOF, models trained on P only), scored on H corrected (production analog)
  (b) k fitted on H corrected, scored on the P teacher (reverse direction, sign check only)
  (c) H cross-fitting with my own splits: 2-fold random halves x 200 seeds, 10-fold x 10 seeds,
      leave-one-region-out over 10 contiguous sub_n blocks
  (d) bootstrap of the (a) gain at H size and at public-LB size, concentration of the gain over rows
  (e) k sensitivity and hard clip for reference
Every effect is reported as change of the TOTAL score (= per-target NRMSE change / 5).
"""
import numpy as np
from scipy.special import log_ndtr

SIG = {"gas": 151750.04, "eui": 40.65}
CAP = 1e6
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH, AP = A[H], A[P]
subnH = z["subn"][H]


def mills(a):
    """phi(a) / Phi(a), stable for very negative a"""
    return np.exp(-0.5 * a * a - 0.5 * np.log(2.0 * np.pi) - log_ndtr(a))


def calib(g, k, cap=CAP):
    """E[y | y < cap] for y ~ N(g, (k g)^2); only rows with g > 5e5 are touched (others change < 1e-12)"""
    out = np.array(g, dtype=float, copy=True)
    if k <= 0:
        return out
    m = out > 5e5
    s = k * out[m]
    out[m] = out[m] - s * mills((cap - out[m]) / s)
    return out


def calib_eui(e, area, k):
    g = e * area
    gc = calib(g, k)
    out = np.array(e, dtype=float, copy=True)
    m = gc != g
    out[m] = e[m] * gc[m] / g[m]
    return out


KGRID = np.round(np.arange(0.0, 0.3001, 0.0025), 4)


def fit_k(p, y, area=None):
    """grid search of k minimising SSE on (p, y); area given -> p, y are EUI and the cap acts on p*area"""
    best, bk = None, 0.0
    m = (p * (area if area is not None else 1.0)) > 5e5
    pm, ym = p[m], y[m]
    am = area[m] if area is not None else None
    for k in KGRID:
        q = calib_eui(pm, am, k) if am is not None else calib(pm, k)
        sse = np.sum((ym - q) ** 2)
        if best is None or sse < best - 1e-9:
            best, bk = sse, k
    return float(bk)


def nrmse(y, p, sig):
    return float(np.sqrt(np.mean((y - p) ** 2)) / sig)


def dtot(y, p0, p1, sig):
    return (nrmse(y, p1, sig) - nrmse(y, p0, sig)) / 5.0


g_y, g_c, g_t = z["gas_y"], z["gas_corrected"], z["gas_teacher"]
g_pP = z["gas_pP"]
g_yP = g_pP + z["gas_rP"]
e_y, e_c = z["eui_y"], z["eui_corrected"]
e_pP = z["eui_pP"]
e_yP = e_pP + z["eui_rP"]
print(f"H rows {len(H)}  P rows {len(P)}")
print(f"base NRMSE on H: gas corrected {nrmse(g_y, g_c, SIG['gas']):.6f}  eui corrected {nrmse(e_y, e_c, SIG['eui']):.6f}")
print(f"truth max on H gas {g_y.max():.1f}, on P gas {g_yP.max():.1f}; H eui*A max {(e_y * AH).max():.1f}")
print(f"H gas corrected > 1e6: {(g_c > CAP).sum()}  > 0.9e6: {(g_c > 9e5).sum()}  > 0.85e6: {(g_c > 8.5e5).sum()}")

# (a) P -> H
kP_gas = fit_k(g_pP, g_yP)
kP_eui = fit_k(e_pP, e_yP, AP)
print(f"\n(a) k fitted on P teacher: gas {kP_gas}  eui(via eui*A) {kP_eui}")
for k_e_label, k_e in (("eui own P-fit k", kP_eui), ("eui with gas P-fit k", kP_gas)):
    dg = dtot(g_y, g_c, calib(g_c, kP_gas), SIG["gas"])
    de = dtot(e_y, e_c, calib_eui(e_c, AH, k_e), SIG["eui"])
    print(f"    H corrected: gas dTotal {dg:+.8f}  eui dTotal {de:+.8f} ({k_e_label})  sum {dg + de:+.8f}")
dg_t = dtot(g_y, g_t, calib(g_t, kP_gas), SIG["gas"])
print(f"    H teacher (no correction): gas dTotal {dg_t:+.8f}")

# (b) reverse: H corrected -> P teacher
kH_gas = fit_k(g_c, g_y)
kH_eui = fit_k(e_c, e_y, AH)
dPg = dtot(g_yP, g_pP, calib(g_pP, kH_gas), SIG["gas"])
dPe = dtot(e_yP, e_pP, calib_eui(e_pP, AP, kH_eui), SIG["eui"])
print(f"\n(b) k fitted on all H corrected: gas {kH_gas}  eui {kH_eui};  scored on P teacher: gas {dPg:+.8f}  eui {dPe:+.8f}")


# (c) cross-fitting on H with my own splits
def crossfit(folds):
    """folds: list of index arrays partitioning H; returns cross-fitted gas and eui predictions"""
    qg, qe = g_c.copy(), e_c.copy()
    ks = []
    for f in folds:
        tr_ = np.ones(len(H), bool)
        tr_[f] = False
        kg = fit_k(g_c[tr_], g_y[tr_])
        ke = fit_k(e_c[tr_], e_y[tr_], AH[tr_])
        qg[f] = calib(g_c[f], kg)
        qe[f] = calib_eui(e_c[f], AH[f], ke)
        ks.append((kg, ke))
    return dtot(g_y, g_c, qg, SIG["gas"]), dtot(e_y, e_c, qe, SIG["eui"]), ks


n = len(H)
res = []
for seed in range(200):
    perm = np.random.default_rng(1000 + seed).permutation(n)
    res.append(crossfit([perm[: n // 2], perm[n // 2:]])[:2])
res = np.array(res)
tot = res.sum(1)
print(f"\n(c1) 2-fold random halves x200 (seeds 1000..1199): gas mean {res[:, 0].mean():+.8f} sd {res[:, 0].std():.8f}"
      f" | eui mean {res[:, 1].mean():+.8f} sd {res[:, 1].std():.8f} | sum mean {tot.mean():+.8f}"
      f" worst {tot.max():+.8f}  share<0 {np.mean(tot < 0):.3f}")
res = []
for seed in range(10):
    perm = np.random.default_rng(5000 + seed).permutation(n)
    res.append(crossfit(np.array_split(perm, 10))[:2])
res = np.array(res)
tot = res.sum(1)
print(f"(c2) 10-fold x10 (seeds 5000..5009): gas mean {res[:, 0].mean():+.8f} sd {res[:, 0].std():.8f}"
      f" | eui mean {res[:, 1].mean():+.8f} | sum mean {tot.mean():+.8f} worst {tot.max():+.8f}")
order = np.argsort(subnH, kind="stable")
blocks = np.array_split(order, 10)  # contiguous sub_n ranges = spatial regions
dg, de, ks = crossfit(blocks)
print(f"(c3) leave-one-region-out (10 contiguous sub_n blocks): gas {dg:+.8f}  eui {de:+.8f}  sum {dg + de:+.8f}")
print(f"     per-block k (gas, eui): {ks}")
# per-region effect with the P-fitted k (is the gain everywhere or in one region?)
qg = calib(g_c, kP_gas)
per = []
for b in blocks:
    per.append((np.sum((g_y[b] - qg[b]) ** 2) - np.sum((g_y[b] - g_c[b]) ** 2)))
per = np.array(per)
print(f"     per-region SSE change (P-fit k, gas, in units of sigma^2): {np.round(per / SIG['gas'] ** 2, 3).tolist()}")

# (d) bootstrap of the P-fit gain, and concentration
qe = calib_eui(e_c, AH, kP_eui)
dg_row = ((g_y - qg) ** 2 - (g_y - g_c) ** 2) / SIG["gas"] ** 2
de_row = ((e_y - qe) ** 2 - (e_y - e_c) ** 2) / SIG["eui"] ** 2
mse_g0 = (g_y - g_c) ** 2 / SIG["gas"] ** 2
mse_e0 = (e_y - e_c) ** 2 / SIG["eui"] ** 2
rng = np.random.default_rng(42)


def boot(size, reps=4000):
    out = np.empty(reps)
    for i in range(reps):
        ix = rng.integers(0, n, size)
        b0g, b0e = mse_g0[ix].mean(), mse_e0[ix].mean()
        out[i] = (np.sqrt(b0g + dg_row[ix].mean()) - np.sqrt(b0g) + np.sqrt(b0e + de_row[ix].mean()) - np.sqrt(b0e)) / 5
    return out


for label, size in (("H size 128791", n), ("public size 55196", 55196)):
    b = boot(size)
    print(f"(d) bootstrap {label}: mean {b.mean():+.8f} sd {b.std():.8f} P(<0) {np.mean(b < 0):.4f}"
          f" P(<=-1e-5) {np.mean(b <= -1e-5):.4f}")
ch = np.flatnonzero(dg_row != 0)
srt = np.sort(dg_row)
print(f"    gas rows changed {len(ch)}; rows that got worse {np.sum(dg_row > 0)} (sum {dg_row[dg_row > 0].sum():.3f} sig^2),"
      f" better {np.sum(dg_row < 0)} (sum {dg_row[dg_row < 0].sum():.3f}); 10 best rows carry"
      f" {srt[:10].sum() / dg_row.sum():.2f} of the net SSE gain; top 10 worst rows {np.sort(dg_row)[-10:].round(3).tolist()}")
dropk = []
for drop in (0, 1, 3, 5, 10):
    keep = np.ones(n, bool)
    keep[np.argsort(dg_row)[:drop]] = False
    d_ = (np.sqrt(mse_g0[keep].mean() + dg_row[keep].mean()) - np.sqrt(mse_g0[keep].mean())) / 5
    dropk.append(f"drop {drop} best rows: {d_:+.8f}")
print("    gas gain robustness: " + "; ".join(dropk))

# (e) sensitivity
print("\n(e) gas / eui dTotal on H corrected for fixed k (and hard clip at 1e6):")
for k in (0.04, 0.06, 0.07, 0.08, 0.085, 0.09, 0.10, 0.12, 0.15):
    print(f"    k {k:5.3f}: gas {dtot(g_y, g_c, calib(g_c, k), SIG['gas']):+.8f}"
          f"  eui {dtot(e_y, e_c, calib_eui(e_c, AH, k), SIG['eui']):+.8f}")
print(f"    hard clip: gas {dtot(g_y, g_c, np.minimum(g_c, CAP), SIG['gas']):+.8f}"
      f"  eui {dtot(e_y, e_c, np.minimum(e_c, CAP / AH), SIG['eui']):+.8f}")
