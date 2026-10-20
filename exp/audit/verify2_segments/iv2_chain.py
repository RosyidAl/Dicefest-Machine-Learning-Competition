"""Independent re-verification (round 'iv') of the gas 1e6 truncated-normal calibration on the holdout chain.

Own code (erfcx Mills ratio, bounded continuous fit of k), honest splits that differ from the claim's:
  S0  k fitted on the P-side teacher OOF, scored on H corrected (the claim's main number, reproduced)
  S1  GroupKFold by subzone on H (5 folds x 20 repeats, seeds 9001..9020)
  S2  three contiguous sub_n regions of H (fit on two, score on the third)
  S3  k fitted on each subzone-parity half of P, scored on H corrected
  S4  reverse: k fitted on all H corrected, scored on the P-side teacher
  S5  fixed-k curve on H corrected; S6 gas + eui combined; S7 concentration of the gain; S8 bootstrap at LB sizes
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_segments/iv2_chain.py
"""
import json
import resource
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erfcx

t0 = time.time()
SIG = {"gas": 151750.04, "eui": 40.65}
CAP = 1e6
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
SC = {"gas": (np.ones(len(H)), np.ones(len(P))), "eui": (A[H], A[P])}  # level in gas units = pred * scale
snH, snP = subn[H], subn[P]
out = {}


def calib(pred, k, scale):
    """Truncated-normal mean below CAP with s = k * level; level = pred * scale; rows with level <= 6e5 unchanged."""
    g = pred * scale
    res = pred.copy()
    m = g > 6e5
    if m.any():
        gm = g[m]
        s = k * gm
        a = (CAP - gm) / s
        gn = gm - s * np.sqrt(2.0 / np.pi) / erfcx(-a / np.sqrt(2.0))
        res[m] = pred[m] * gn / gm
    return res


def fit_k(y, pred, scale):
    m = pred * scale > 6e5
    yy, pp, ss = y[m], pred[m], scale[m]

    def f(k):
        return float(np.sum((yy - calib(pp, k, ss)) ** 2))

    grid = np.arange(0.005, 0.3001, 0.005)
    i = int(np.argmin([f(k) for k in grid]))
    r = minimize_scalar(f, bounds=(max(grid[i] - 0.005, 1e-3), grid[i] + 0.005), method="bounded",
                        options={"xatol": 1e-6})
    return float(r.x)


def dtot(y, base, new, met):
    n = len(y)
    return (np.sqrt(np.sum((y - new) ** 2) / n) - np.sqrt(np.sum((y - base) ** 2) / n)) / SIG[met] / 5


def bias_table(y, pred, scale, label):
    g = pred * scale
    r = (y - pred) * scale
    edges = [0.8, 0.85, 0.9, 0.95, 1.0, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g >= lo * 1e6) & (g < hi * 1e6)
        n = int(m.sum())
        mu = r[m].mean() if n else np.nan
        t = mu / (r[m].std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
        rows.append(f"[{lo:.2f},{hi:.2f}) n {n:4d} mean {mu:+9.0f} t {t:+5.2f}")
    print(f"  {label}: " + " | ".join(rows))


for met in ("gas", "eui"):
    y, c, tch = z[f"{met}_y"], z[f"{met}_corrected"], z[f"{met}_teacher"]
    pP = z[f"{met}_pP"]
    yP = pP + z[f"{met}_rP"]
    sH, sP = SC[met]
    nr = np.sqrt(np.mean((y - c) ** 2)) / SIG[met]
    print(f"\n===== {met}: H corrected NRMSE {nr:.6f}  (H n {len(H)}, P n {len(P)})")
    print("bias of residual (gas units) by level = pred*scale / 1e6:")
    bias_table(yP, pP, sP, "P teacher  ")
    bias_table(y, c, sH, "H corrected")
    o = out[met] = {}
    # S0
    kP = fit_k(yP, pP, sP)
    o["S0_kP"] = kP
    o["S0_H_corrected"] = dtot(y, c, calib(c, kP, sH), met)
    o["S0_H_teacher"] = dtot(y, tch, calib(tch, kP, sH), met)
    print(f"S0 k fitted on P teacher {kP:.4f} -> H corrected dTotal {o['S0_H_corrected']:+.8f}  (H teacher {o['S0_H_teacher']:+.8f})")
    bias_table(y, calib(c, kP, sH), sH, "H corr. after S0 calib")
    # S1 GroupKFold by subzone on H
    reps = []
    ks = []
    for seed in range(9001, 9021):
        rng = np.random.default_rng(seed)
        u = np.unique(snH)
        fold_of = dict(zip(u, rng.integers(0, 5, len(u))))
        f = np.vectorize(fold_of.get)(snH) if seed == 9001 else None
        if f is None:
            lab = rng.integers(0, 5, len(u))  # fresh draw for later seeds (cheap path)
            f = lab[np.searchsorted(u, snH)]
        new = c.copy()
        for j in range(5):
            te_ = f == j
            k = fit_k(y[~te_], c[~te_], sH[~te_])
            ks.append(k)
            new[te_] = calib(c[te_], k, sH[te_])
        reps.append(dtot(y, c, new, met))
    reps = np.array(reps)
    o["S1_group5x20"] = dict(mean=float(reps.mean()), sd=float(reps.std(ddof=1)), worst=float(reps.max()),
                             k_min=float(min(ks)), k_max=float(max(ks)))
    print(f"S1 GroupKFold(subzone) 5x20 on H: dTotal mean {reps.mean():+.8f} sd {reps.std(ddof=1):.8f} worst {reps.max():+.8f}"
          f"  k range {min(ks):.4f}..{max(ks):.4f}")
    # S2 contiguous sub_n thirds of H
    q = np.quantile(snH, [1 / 3, 2 / 3])
    blk = np.digitize(snH, q)
    new = c.copy()
    per = []
    for j in range(3):
        te_ = blk == j
        k = fit_k(y[~te_], c[~te_], sH[~te_])
        new[te_] = calib(c[te_], k, sH[te_])
        dsse = (np.sum((y[te_] - new[te_]) ** 2) - np.sum((y[te_] - c[te_]) ** 2)) / SIG[met] ** 2
        per.append((round(k, 4), round(float(dsse), 3), int(te_.sum())))
    o["S2_thirds"] = dtot(y, c, new, met)
    print(f"S2 contiguous thirds of H: dTotal {o['S2_thirds']:+.8f}  per block (k, dSSE/sigma^2, n): {per}")
    # S3 P halves by subzone parity -> H
    s3 = []
    for par in (0, 1):
        mP = (snP % 2) == par
        k = fit_k(yP[mP], pP[mP], sP[mP])
        s3.append((round(k, 4), dtot(y, c, calib(c, k, sH), met)))
    o["S3_Phalves"] = s3
    print(f"S3 k from P half (subzone parity 0 / 1) -> H corrected: " + "; ".join(f"k {k:.4f} dTotal {d:+.8f}" for k, d in s3))
    # S4 reverse
    kH = fit_k(y, c, sH)
    o["S4_kH"] = kH
    o["S4_P_teacher"] = dtot(yP, pP, calib(pP, kH, sP), met)
    print(f"S4 k fitted on all H corrected {kH:.4f} -> P teacher dTotal {o['S4_P_teacher']:+.8f}")
    # S5 fixed k on H corrected
    curve = {f"{k:.3f}": dtot(y, c, calib(c, k, sH), met) for k in (0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14)}
    o["S5_curve_H"] = curve
    print("S5 fixed k on H corrected: " + "  ".join(f"{k} {v:+.8f}" for k, v in curve.items()))
    curveP = {f"{k:.3f}": dtot(yP, pP, calib(pP, k, sP), met) for k in (0.05, 0.08, 0.10, 0.12)}
    print("   fixed k on P teacher : " + "  ".join(f"{k} {v:+.8f}" for k, v in curveP.items()))
    # S7 concentration (S0 k on H corrected)
    new = calib(c, kP, sH)
    d = ((y - new) ** 2 - (y - c) ** 2) / SIG[met] ** 2
    ch = np.abs(new - c) > 1e-9 * np.abs(c)
    srt = np.sort(d[ch])
    o["S7"] = dict(rows_changed=int(ch.sum()), better=int((d < 0).sum()), worse=int((d > 0).sum()),
                   sum_better=float(d[d < 0].sum()), sum_worse=float(d[d > 0].sum()))
    big = np.abs(d) > 0.01
    print(f"S7 rows changed {ch.sum()}; better {(d < 0).sum()} (sum {d[d < 0].sum():.3f} sigma^2), worse {(d > 0).sum()} "
          f"(sum {d[d > 0].sum():.3f}); rows with |dSE| > 0.01 sigma^2: {big.sum()} of which better {(d[big] < 0).sum()}")
    print(f"   10 largest gains {np.round(srt[:10], 3).tolist()}  10 largest losses {np.round(srt[-10:], 3).tolist()}")
    n = len(y)
    for tr_ in (1, 3, 5):
        dd = np.sort(d)
        keep = np.ones(n, bool)
        idx = np.argsort(d)
        keep[idx[:tr_]] = False
        keep[idx[-tr_:]] = False
        tot = np.sqrt((np.sum((y[keep] - c[keep]) ** 2) / SIG[met] ** 2 + d[keep].sum()) / keep.sum()) - \
            np.sqrt(np.sum((y[keep] - c[keep]) ** 2) / SIG[met] ** 2 / keep.sum())
        print(f"   symmetric trim (drop {tr_} best and {tr_} worst rows): dTotal {tot / 5:+.8f}")
    # S8 bootstrap at LB sizes with the S0 k
    rng = np.random.default_rng(31337)
    base_se = (y - c) ** 2 / SIG[met] ** 2
    for name, size in (("public", 55196), ("private", 128791)):
        bs = []
        for _ in range(4000):
            ii = rng.integers(0, n, size)
            b = base_se[ii].mean()
            bs.append((np.sqrt(b + d[ii].mean()) - np.sqrt(b)) / 5)
        bs = np.array(bs)
        o[f"S8_{name}"] = dict(mean=float(bs.mean()), sd=float(bs.std()), p_improve=float((bs < 0).mean()))
        print(f"S8 bootstrap {name} size {size}: mean {bs.mean():+.8f} sd {bs.std():.8f} P(improve) {(bs < 0).mean():.4f}")
    o["d_rows"] = d  # kept for the combined block below (not dumped)

# S6 combined gas + eui
print("\n===== combined gas + eui on H corrected")
g = out["gas"]
e = out["eui"]
print(f"S0 (each metric's own P-fit k): {g['S0_H_corrected'] + e['S0_H_corrected']:+.8f}")
yg, cg = z["gas_y"], z["gas_corrected"]
ye, ce = z["eui_y"], z["eui_corrected"]
dg = dtot(yg, cg, calib(cg, 0.08, SC["gas"][0]), "gas")
de = dtot(ye, ce, calib(ce, 0.08, SC["eui"][0]), "eui")
print(f"k = 0.08 both (candidate): gas {dg:+.8f} eui {de:+.8f} sum {dg + de:+.8f}")
de_pg = dtot(ye, ce, calib(ce, g["S0_kP"], SC["eui"][0]), "eui")
print(f"eui with the gas P-fit k {g['S0_kP']:.4f}: {de_pg:+.8f}")
# public-size bootstrap of the combined change (same resampled rows for both metrics)
rng = np.random.default_rng(4711)
bg = (yg - cg) ** 2 / SIG["gas"] ** 2
be = (ye - ce) ** 2 / SIG["eui"] ** 2
dgr, der = g["d_rows"], e["d_rows"]
for name, size in (("public", 55196), ("private", 128791)):
    bs = []
    for _ in range(4000):
        ii = rng.integers(0, len(yg), size)
        b1, b2 = bg[ii].mean(), be[ii].mean()
        bs.append((np.sqrt(b1 + dgr[ii].mean()) - np.sqrt(b1) + np.sqrt(b2 + der[ii].mean()) - np.sqrt(b2)) / 5)
    bs = np.array(bs)
    print(f"combined bootstrap {name}: mean {bs.mean():+.8f} sd {bs.std():.8f} P(improve) {(bs < 0).mean():.4f}")
for met in out:
    out[met].pop("d_rows")
json.dump(out, open("exp/audit/verify2_segments/iv2_chain.json", "w"), indent=1)
print(f"\nelapsed {time.time() - t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
