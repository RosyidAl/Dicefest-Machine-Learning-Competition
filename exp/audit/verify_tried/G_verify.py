"""Adversarial re-derivation of the 'G' gas-cap fix claimed by the 'tried' audit lens.

Claim: gas is hard-truncated below 1e6 kWh; shrink predictions above a knot (9e5):
    g' = min(g + beta * max(g - 9e5, 0), 1e6), beta fitted on P (teacher-level pP vs yP);
    EUI via the implied gas e*A: e' = G(e*A)/A.
Claimed honest total change on the holdout chain: -0.0000234.

This script (holdout chain only, exp/audit/chain_H.npz; no raw data):
  1. re-derives the key number from scratch (beta on P, scored on the corrected H predictions);
  2. scale / identity checks (gas_y == eui_y * A, sigmas, total = sum / 5);
  3. H-blind knot choice (5-fold CV inside P over a knot grid), then scored on H;
  4. joint bootstrap (resample P to refit beta AND resample H to score);
  5. influence / jackknife (how much rests on the few most-helped rows);
  6. teacher vs corrected top-bin bias on H (does the kNN correction absorb the bias? -> test transfer);
  7. placebo: the same hinge machinery on elec (not truncated) at a knot with the same tail share.
Writes nothing except its own log (stdout)."""
import numpy as np
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
CAP, KNOT = 1e6, 9e5
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AP, AH = A[P], A[H]
g = dict(y=z["gas_y"], c=z["gas_corrected"], t=z["gas_teacher"], pP=z["gas_pP"], rP=z["gas_rP"])
e = dict(y=z["eui_y"], c=z["eui_corrected"], t=z["eui_teacher"], pP=z["eui_pP"], rP=z["eui_rP"])
el = dict(y=z["elec_y"], c=z["elec_corrected"], pP=z["elec_pP"], rP=z["elec_rP"])
g["yP"] = g["pP"] + g["rP"]
e["yP"] = e["pP"] + e["rP"]
el["yP"] = el["pP"] + el["rP"]
nH = len(H)

print("== 1. identities / scales")
print(f"  |P|={len(P)} |H|={nH} overlap={len(np.intersect1d(P, H))}")
print(f"  max |gas_y - eui_y*A_H| / gas_y = {np.max(np.abs(g['y'] - e['y'] * AH) / g['y']):.2e}")
print(f"  max |gas_yP - eui_yP*A_P| / gas_yP = {np.max(np.abs(g['yP'] - e['yP'] * AP) / g['yP']):.2e}")
print(f"  y max: P {g['yP'].max():.1f}  H {g['y'].max():.1f}  (cap {CAP:.0f}); rows >= cap P {(g['yP'] >= CAP).sum()} H {(g['y'] >= CAP).sum()}")


def nrmse(y, p, m):
    return np.sqrt(np.mean((y - p) ** 2)) / SIG[m]


b0g, b0e = nrmse(g["y"], g["c"], "gas"), nrmse(e["y"], e["c"], "eui")
print(f"  baseline corrected H: gas {b0g:.6f} eui {b0e:.6f} (claimed 0.24514 / 0.64859)")


def hinge_beta(p, y, knot=KNOT):
    h = np.maximum(p - knot, 0)
    return float(h @ (y - p) / (h @ h))


def G(v, beta, knot=KNOT):
    return np.minimum(v + beta * np.maximum(v - knot, 0), CAP)


def apply_all(beta, knot=KNOT, cg=None, ce=None):
    cg = g["c"] if cg is None else cg
    ce = e["c"] if ce is None else ce
    fg = G(cg, beta, knot)
    ge = ce * AH
    fe = np.where(ge > knot, G(ge, beta, knot) / AH, ce)  # untouched rows stay bit-identical
    return fg, fe


def total_change(fg, fe, cg=None, ce=None):
    cg = g["c"] if cg is None else cg
    ce = e["c"] if ce is None else ce
    dg = nrmse(g["y"], fg, "gas") - nrmse(g["y"], cg, "gas")
    de = nrmse(e["y"], fe, "eui") - nrmse(e["y"], ce, "eui")
    return dg, de, (dg + de) / 5


print("\n== 2. re-derivation of the claimed number")
beta = hinge_beta(g["pP"], g["yP"])
fg, fe = apply_all(beta)
dg, de, tot = total_change(fg, fe)
print(f"  beta on P (teacher pP vs yP, knot 9e5): {beta:.4f}  (claimed -0.7275); P rows above knot {(g['pP'] > KNOT).sum()}")
print(f"  gas {dg:+.7f}  eui {de:+.7f}  TOTAL {tot:+.8f}  (claimed gas -0.0000936, eui -0.0000234, total -0.0000234)")
print(f"  rows changed: gas {(fg != g['c']).sum()}  eui {(fe != e['c']).sum()}")
# alternative: beta fitted on P with intercept-free hinge on the CORRECTED-like level is unavailable (P has no
# cross-fitted correction); check the teacher-level effect on H as well
ft, fte = apply_all(beta, cg=g["t"], ce=e["t"])
dgt, det, tott = total_change(ft, fte, cg=g["t"], ce=e["t"])
print(f"  same beta applied to the H TEACHER (no residual correction): gas {dgt:+.7f} eui {det:+.7f} TOTAL {tott:+.8f}")

print("\n== 3. H-blind knot choice: 5-fold CV inside P over a knot grid, chosen knot + beta refit on all P, scored on H")
grid = [6e5, 7e5, 7.5e5, 8e5, 8.5e5, 8.75e5, 9e5, 9.25e5, 9.5e5]
kfP = list(KFold(5, shuffle=True, random_state=1).split(np.zeros(len(P))))
cvP = {}
for k in grid + [None]:
    sse = 0.0
    for a, b in kfP:
        if k is None:  # hard cap only
            q = np.minimum(g["pP"][b], CAP)
        else:
            bt = hinge_beta(g["pP"][a], g["yP"][a], k)
            q = G(g["pP"][b], bt, k)
        sse += np.sum((g["yP"][b] - q) ** 2) - np.sum((g["yP"][b] - g["pP"][b]) ** 2)
    cvP[k] = sse / len(P)
for k, v in cvP.items():
    lab = "hardcap" if k is None else f"{k:.3g}"
    print(f"  knot {lab:>8}: P-CV dMSE {v:+.1f}")
kbest = min((k for k in grid), key=lambda k: cvP[k])
bb = hinge_beta(g["pP"], g["yP"], kbest)
fg3, fe3 = apply_all(bb, kbest)
d3 = total_change(fg3, fe3)
print(f"  chosen on P: knot {kbest:.3g} beta {bb:.4f} -> H gas {d3[0]:+.7f} eui {d3[1]:+.7f} TOTAL {d3[2]:+.8f}")
for k in grid:
    bk = hinge_beta(g["pP"], g["yP"], k)
    dk = total_change(*apply_all(bk, k))
    print(f"    (all knots, for shape) knot {k:.3g} beta {bk:+.4f}: H TOTAL {dk[2]:+.8f}")

print("\n== 4. joint bootstrap: resample P (beta refit) and H (scoring), 1000 reps")
rng = np.random.default_rng(12345)
selP = np.where(g["pP"] > KNOT)[0]
hP = g["pP"][selP] - KNOT
rPsel = g["yP"][selP] - g["pP"][selP]
eg0 = (g["y"] - g["c"]) ** 2
ee0 = (e["y"] - e["c"]) ** 2
chg = np.where(g["c"] > KNOT)[0]
geH = e["c"] * AH
che = np.where(geH > KNOT)[0]
tots, betas = [], []
for _ in range(1000):
    wP = np.bincount(rng.integers(0, len(P), len(P)), minlength=len(P))[selP]
    bt = float((wP * hP) @ rPsel / ((wP * hP) @ hP))
    wH = np.bincount(rng.integers(0, nH, nH), minlength=nH).astype(float)
    eg1 = eg0.copy()
    eg1[chg] = (g["y"][chg] - G(g["c"][chg], bt)) ** 2
    ee1 = ee0.copy()
    ee1[che] = (e["y"][che] - G(geH[che], bt) / AH[che]) ** 2
    m0g, m1g = wH @ eg0 / nH, wH @ eg1 / nH
    m0e, m1e = wH @ ee0 / nH, wH @ ee1 / nH
    tots.append(((np.sqrt(m1g) - np.sqrt(m0g)) / SIG["gas"] + (np.sqrt(m1e) - np.sqrt(m0e)) / SIG["eui"]) / 5)
    betas.append(bt)
tots, betas = np.array(tots), np.array(betas)
print(f"  TOTAL mean {tots.mean():+.8f} sd {tots.std():.8f}  P(<0) {(tots < 0).mean():.3f}  5%..95% "
      f"[{np.quantile(tots, .05):+.8f}, {np.quantile(tots, .95):+.8f}]")
print(f"  beta bootstrap: mean {betas.mean():.3f} sd {betas.std():.3f} 5..95% [{np.quantile(betas, .05):.3f}, {np.quantile(betas, .95):.3f}]")

print("\n== 5. influence: per-row squared-error change on H (gas), share of the gain from the most-helped rows")
dsq = (g["y"] - fg) ** 2 - eg0  # negative = helped
o = np.argsort(dsq)
tot_dsq = dsq.sum()
for k in (1, 3, 5, 10, 20):
    print(f"  top-{k:2d} helped rows carry {dsq[o[:k]].sum() / tot_dsq:6.1%} of the gas SSE reduction")
print(f"  rows hurt: {(dsq > 0).sum()} of {(dsq != 0).sum()} changed; hurt SSE share {dsq[dsq > 0].sum() / -tot_dsq:.1%} of the net gain")
for drop in (1, 3, 5, 10):
    keep = np.ones(nH, bool)
    keep[o[:drop]] = False
    dgd = (np.sqrt(np.mean(((g["y"] - fg) ** 2)[keep])) - np.sqrt(np.mean(eg0[keep]))) / SIG["gas"]
    dged = (np.sqrt(np.mean(((e["y"] - fe) ** 2)[keep])) - np.sqrt(np.mean(ee0[keep]))) / SIG["eui"]
    print(f"  drop the {drop:2d} most-helped gas rows from H: gas {dgd:+.7f} eui {dged:+.7f} TOTAL {(dgd + dged) / 5:+.8f}")
# leave-one-out over the changed rows: worst case
loo = []
for i in np.where(dsq != 0)[0]:
    loo.append(dsq.sum() - dsq[i])
print(f"  LOO over changed rows: max remaining SSE change {max(loo):+.3e} (negative = still a gain)")

print("\n== 6. bias by prediction bin on H: teacher vs corrected, and on P (teacher)")
for lo, hi in ((7e5, 8e5), (8e5, 8.5e5), (8.5e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 1e8)):
    st = (g["t"] >= lo) & (g["t"] < hi)
    sc = (g["c"] >= lo) & (g["c"] < hi)
    sp = (g["pP"] >= lo) & (g["pP"] < hi)
    print(f"  [{lo:.3g},{hi:.3g}): H teacher n={st.sum():4d} mean r {np.mean(g['y'][st] - g['t'][st]) if st.any() else np.nan:+9.0f} | "
          f"H corrected n={sc.sum():4d} mean r {np.mean(g['y'][sc] - g['c'][sc]) if sc.any() else np.nan:+9.0f} | "
          f"P n={sp.sum():4d} mean r {np.mean(g['rP'][sp]) if sp.any() else np.nan:+9.0f}")
s = g["c"] > KNOT
print(f"  rows c>9e5: correction (c - t) mean {np.mean(g['c'][s] - g['t'][s]):+.0f}, mean resid teacher "
      f"{np.mean(g['y'][s] - g['t'][s]):+.0f} corrected {np.mean(g['y'][s] - g['c'][s]):+.0f}")

print("\n== 7. placebo on elec (no truncation): hinge at the elec P-prediction quantile with the same tail share")
share = (g["pP"] > KNOT).mean()
kel = float(np.quantile(el["pP"], 1 - share))
bel = hinge_beta(el["pP"], el["yP"], kel)
fel = el["c"] + bel * np.maximum(el["c"] - kel, 0)
dl = nrmse(el["y"], fel, "elec") - nrmse(el["y"], el["c"], "elec")
print(f"  elec knot {kel:.0f} (tail share {share:.5f}), beta on P {bel:+.4f}, H elec change {dl:+.7f} total {dl / 5:+.8f}")
for q in (0.999, 0.9995):
    kq = float(np.quantile(el["pP"], q))
    bq = hinge_beta(el["pP"], el["yP"], kq)
    fq = el["c"] + bq * np.maximum(el["c"] - kq, 0)
    dq = nrmse(el["y"], fq, "elec") - nrmse(el["y"], el["c"], "elec")
    print(f"  elec knot q{q} = {kq:.0f}: beta {bq:+.4f} H total {dq / 5:+.8f}")
# placebo on gas BELOW the cap region: tent basis on [7e5, 8.75e5] (zero outside; no truncation effect expected there)
lo_, hi_ = 7e5, 8.75e5


def tent(v):
    return np.maximum(0, np.minimum(v - lo_, hi_ - v))


tP = tent(g["pP"])
bband = float(tP @ (g["yP"] - g["pP"]) / (tP @ tP))
db = nrmse(g["y"], g["c"] + bband * tent(g["c"]), "gas") - b0g
print(f"  gas placebo tent [7e5, 8.75e5] slope on P {bband:+.4f}: H gas {db:+.7f} total {db / 5:+.8f} "
      f"(rows {(tent(g['c']) > 0).sum()})")
