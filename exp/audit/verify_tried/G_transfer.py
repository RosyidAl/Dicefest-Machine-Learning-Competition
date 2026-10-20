"""Transfer check of the 'G' gas-cap fix to the test set (adversarial verifier).

1. Distribution of the submitted gas predictions (sub_v19vce, after correction + dup copy) and of the implied EUI gas
   e*A near the cap, vs the corrected holdout-chain predictions on H (scaled to the test size).
2. Expected test gain: (a) parametric, under the P-fitted bias model b(c) = beta*(c - knot) the expected SSE change of a
   row is -beta^2 h^2 (h = c - knot, ignoring the 1e6 cap), so the expected gain scales with mean(h^2);
   (b) the realised H gain per row, reweighted by prediction bin to the test bin counts.
3. Independent full-train OOF check (deployed v19 gas members, global NNLS weights from summary.json, teacher level):
   G only, beta cross-fitted 5-fold and 2-fold over train rows. Bias per bin.
Light: column subsets only."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65}
CAP, KNOT = 1e6, 9e5
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH = A[H]
yH, cH, yHe, cHe = z["gas_y"], z["gas_corrected"], z["eui_y"], z["eui_corrected"]
pP, yP = z["gas_pP"], z["gas_pP"] + z["gas_rP"]
h = np.maximum(pP - KNOT, 0)
beta = float(h @ (yP - pP) / (h @ h))

sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["id", GAS, EUI], float_precision="round_trip")
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS, EUI])
assert (sub["id"].values == te["id"].values).all()
dup = te[KEY].merge(tr[KEY].drop_duplicates().assign(_d=1), on=KEY, how="left")["_d"].notna().values
At = te["bld_dim_06"].values
sg, se = sub[GAS].values, sub[EUI].values
nT, nH = len(sg), len(H)
print(f"beta(P) {beta:.4f}; test rows {nT}, dup rows {dup.sum()}, dup rows with gas > 9e5: {(dup & (sg > KNOT)).sum()}")

print("\n== 1. counts per prediction bin (H scaled to the test size)")
edges = [8e5, 8.5e5, 9e5, 9.25e5, 9.5e5, 9.75e5, 1e6, 1.03e6, 1e8]
cT, _ = np.histogram(sg[~dup], edges)
cHh, _ = np.histogram(cH, edges)
for i in range(len(edges) - 1):
    print(f"  gas [{edges[i]:.4g},{edges[i + 1]:.4g}): test {cT[i]:4d} | H {cHh[i]:4d} -> scaled {cHh[i] * nT / nH:6.1f}")
geT, geH = se * At, cHe * AH
cTe, _ = np.histogram(geT[~dup], edges)
cHe_, _ = np.histogram(geH, edges)
print("  implied EUI gas e*A:")
for i in range(len(edges) - 1):
    print(f"  e*A [{edges[i]:.4g},{edges[i + 1]:.4g}): test {cTe[i]:4d} | H {cHe_[i]:4d} -> scaled {cHe_[i] * nT / nH:6.1f}")

print("\n== 2a. parametric expected gain: mean(h^2) test vs H (h = max(c - 9e5, 0), capped shrink ignored)")
hT = np.where(dup, 0, np.maximum(sg - KNOT, 0))
hH = np.maximum(cH - KNOT, 0)
mT, mH = np.mean(hT ** 2), np.mean(hH ** 2)
print(f"  gas  mean h^2: test {mT:.4e}  H {mH:.4e}  ratio test/H {mT / mH:.3f}")
heT = np.where(dup, 0, np.maximum(geT - KNOT, 0))
heH = np.maximum(geH - KNOT, 0)
# eui expected gain uses (h/A)^2
meT, meH = np.mean((heT / At) ** 2), np.mean((heH / AH) ** 2)
print(f"  eui  mean (h/A)^2: test {meT:.4e}  H {meH:.4e}  ratio test/H {meT / meH:.3f}")
# model-based expected total change (first order, dRMSE = dMSE / (2 RMSE0)), using H's RMSE0
r0g, r0e = np.sqrt(np.mean((yH - cH) ** 2)), np.sqrt(np.mean((yHe - cHe) ** 2))
eg_H, eg_T = -beta ** 2 * mH / (2 * r0g) / SIG["gas"], -beta ** 2 * mT / (2 * r0g) / SIG["gas"]
ee_H, ee_T = -beta ** 2 * meH / (2 * r0e) / SIG["eui"], -beta ** 2 * meT / (2 * r0e) / SIG["eui"]
print(f"  model-expected TOTAL on H {(eg_H + ee_H) / 5:+.8f} (realised H -0.00002341); on test {(eg_T + ee_T) / 5:+.8f}")


def G(v, b=beta):
    return np.minimum(v + b * np.maximum(v - KNOT, 0), CAP)


print("\n== 2b. realised H per-row SSE change, reweighted to the test bin counts")
fg = G(cH)
dsq = (yH - fg) ** 2 - (yH - cH) ** 2
ge = G(geH) / AH
dsqe = np.where(geH > KNOT, (yHe - ge) ** 2 - (yHe - cHe) ** 2, 0.0)
e2 = [9e5, 9.5e5, 1e6, 1e8]
tg = te_ = 0.0
for i in range(len(e2) - 1):
    sH = (cH >= e2[i]) & (cH < e2[i + 1])
    sT = (sg >= e2[i]) & (sg < e2[i + 1]) & ~dup
    sHe = (geH >= e2[i]) & (geH < e2[i + 1])
    sTe = (geT >= e2[i]) & (geT < e2[i + 1]) & ~dup
    mg = dsq[sH].mean() if sH.any() else 0.0
    me = dsqe[sHe].mean() if sHe.any() else 0.0
    tg += mg * sT.sum()
    te_ += me * sTe.sum()
    print(f"  bin [{e2[i]:.3g},{e2[i + 1]:.3g}): H n={sH.sum():3d} mean dSSE {mg:+.3e} | test n={sT.sum():3d}"
          f" || eui H n={sHe.sum():3d} mean dSSE {me:+.3e} | test n={sTe.sum():3d}")
dT = (tg / nT / (2 * r0g)) / SIG["gas"] + (te_ / nT / (2 * r0e)) / SIG["eui"]
dHr = (dsq.sum() / nH / (2 * r0g)) / SIG["gas"] + (dsqe.sum() / nH / (2 * r0e)) / SIG["eui"]
print(f"  first-order TOTAL: H {dHr / 5:+.8f}; test (bin-reweighted) {dT / 5:+.8f}")

print("\n== 3. full-train OOF, deployed v19 gas members, global NNLS weights (teacher level)")
s = json.load(open("final/work/v19vce/summary.json"))
At_tr = tr["bld_dim_06"].values
yg = tr[GAS].values
p = sum(np.load(f"exp/oof/{k}.npy") * v for k, v in s["weights"]["gas"].items() if v > 0) * At_tr
pe = sum(np.load(f"exp/oof/{k}.npy") * v for k, v in s["weights"]["eui"].items() if v > 0)
ye = tr[EUI].values
print(f"  OOF gas NRMSE {np.sqrt(np.mean((yg - p) ** 2)) / SIG['gas']:.6f}  eui {np.sqrt(np.mean((ye - pe) ** 2)) / SIG['eui']:.6f}")
for lo, hi in ((8e5, 8.5e5), (8.5e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 1e8)):
    sl = (p >= lo) & (p < hi)
    print(f"  pred [{lo:.3g},{hi:.3g}): n={sl.sum():4d} mean resid {np.mean(yg[sl] - p[sl]):+9.0f}")
hh = np.maximum(p - KNOT, 0)
print(f"  beta on all train OOF {float(hh @ (yg - p) / (hh @ hh)):.4f}; rows > 9e5: {(p > KNOT).sum()} "
      f"(per 1e5 rows {1e5 * (p > KNOT).mean():.1f}; H corrected {1e5 * (cH > KNOT).mean():.1f}; test {1e5 * (sg > KNOT).mean():.1f})")
for nf in (5, 2):
    tot = []
    for seed in range(5):
        qg, qe = p.copy(), pe.copy()
        for a, b in KFold(nf, shuffle=True, random_state=seed).split(p):
            ha = np.maximum(p[a] - KNOT, 0)
            bt = float(ha @ (yg[a] - p[a]) / (ha @ ha))
            qg[b] = G(p[b], bt)
            gb = pe[b] * At_tr[b]
            qe[b] = np.where(gb > KNOT, G(gb, bt) / At_tr[b], pe[b])
        dg = (np.sqrt(np.mean((yg - qg) ** 2)) - np.sqrt(np.mean((yg - p) ** 2))) / SIG["gas"]
        de = (np.sqrt(np.mean((ye - qe) ** 2)) - np.sqrt(np.mean((ye - pe) ** 2))) / SIG["eui"]
        tot.append((dg, de, (dg + de) / 5))
    tot = np.array(tot)
    print(f"  {nf}-fold cross-fit (5 seeds): gas {tot[:, 0].mean():+.7f} eui {tot[:, 1].mean():+.7f} "
          f"TOTAL {tot[:, 2].mean():+.8f} (seed range {tot[:, 2].min():+.8f}..{tot[:, 2].max():+.8f})")
# beta from the chain's P applied to full OOF rows that are H rows only vs P rows only (P rows are in-sample for beta)
qg = G(p)
for nm, idx in (("H rows", H), ("P rows", P)):
    dg = (np.sqrt(np.mean((yg[idx] - qg[idx]) ** 2)) - np.sqrt(np.mean((yg[idx] - p[idx]) ** 2))) / SIG["gas"]
    print(f"  chain-P beta applied to full-OOF {nm}: gas {dg:+.7f} (gas only, total {dg / 5:+.8f})")
