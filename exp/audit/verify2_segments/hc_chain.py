"""verify2 (segments lens): honest re-measurement of the hard clip gas <= 1e6 (and EUI <= 1e6/A).

Inputs: exp/audit/chain_H.npz only (no raw data). Own code, independent of exp/audit/segments/seg_bounds.py.
Evaluations (no parameter is fitted on the scored rows):
  E1  H rows, corrected (honest analog of the submission), bound = 1e6 (fixed) and bound = max gas over P rows.
  E2  P rows, teacher pP (P-OOF member predictions, VC blend), bound = 1e6 and bound = max gas over H rows (other part).
  E3  H re-split with new seeds: 2 halves (seed 20261020) and 5 folds (seed 123); bound for each part from the
      complement (P + other H parts); per-part dTotal.
  E4  bootstrap sd of the H effect (2000 resamples of H rows, seed 99).
Every effect is reported as dNRMSE of the metric and dTotal = dNRMSE / 5.
"""
import json
import resource

import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65}
CAP = 1e6
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH, AP = A[H], A[P]
yH = {m: z[f"{m}_y"] for m in SIG}
cH = {m: z[f"{m}_corrected"] for m in SIG}
tH = {m: z[f"{m}_teacher"] for m in SIG}
pP = {m: z[f"{m}_pP"] for m in SIG}
yP = {m: z[f"{m}_pP"] + z[f"{m}_rP"] for m in SIG}
out = {}

# sanity: gas == eui * A on both parts; base scores match the chain numbers
print(f"check gas==eui*A  H max rel {np.max(np.abs(yH['gas'] - yH['eui'] * AH) / yH['gas']):.2e}  "
      f"P max rel {np.max(np.abs(yP['gas'] - yP['eui'] * AP) / yP['gas']):.2e}")
print(f"y max: H gas {yH['gas'].max():.1f}  P gas {yP['gas'].max():.1f}  (rows >= 1e6: H {np.sum(yH['gas'] >= CAP)}, "
      f"P {np.sum(yP['gas'] >= CAP)})")


def nr(y, p, m):
    return np.sqrt(np.mean((y - p) ** 2)) / SIG[m]


def clip_effect(y, p, cap_vec, m):
    """dNRMSE of min(p, cap) vs p, plus row-level detail"""
    q = np.minimum(p, cap_vec)
    ch = q < p
    d = nr(y, q, m) - nr(y, p, m)
    dse = (q - y) ** 2 - (p - y) ** 2  # per-row change of squared error
    return dict(n=int(ch.sum()), base=float(nr(y, p, m)), dnrmse=float(d), dtotal=float(d / 5),
                rows_better=int(np.sum(dse[ch] < 0)), rows_worse=int(np.sum(dse[ch] > 0)),
                y_of_changed_max=float(y[ch].max()) if ch.any() else None)


def line(tag, r):
    print(f"  {tag:<52s} rows {r['n']:3d} (better {r['rows_better']:3d} worse {r['rows_worse']:2d})  base {r['base']:.5f}  "
          f"dNRMSE {r['dnrmse']:+.7f}  dTotal {r['dtotal']:+.8f}")


print("\nE1  H rows, corrected prediction (honest analog of sub_v19vce)")
bP = yP["gas"].max()
for tag, cg in [("gas clip 1e6", CAP), (f"gas clip max gas over P ({bP:.0f})", bP)]:
    r = clip_effect(yH["gas"], cH["gas"], np.full(len(H), cg), "gas"); line(tag, r); out[f"E1 {tag}"] = r
for tag, cg in [("eui clip 1e6/A", CAP), (f"eui clip maxgasP/A", bP)]:
    r = clip_effect(yH["eui"], cH["eui"], cg / AH, "eui"); line(tag, r); out[f"E1 {tag}"] = r
g1 = out["E1 gas clip 1e6"]["dtotal"] + out["E1 eui clip 1e6/A"]["dtotal"]
print(f"  E1 gas+eui (1e6, 1e6/A) dTotal {g1:+.8f}")
print("  teacher instead of corrected (for reference):")
for tag, m, cv in [("gas clip 1e6 (teacher)", "gas", np.full(len(H), CAP)), ("eui clip 1e6/A (teacher)", "eui", CAP / AH)]:
    r = clip_effect(yH[m], tH[m], cv, m); line(tag, r); out[f"E1 {tag}"] = r

print("\nE2  P rows (disjoint from H), teacher pP from P-OOF members")
bH = yH["gas"].max()
for tag, cg in [("gas clip 1e6", CAP), (f"gas clip max gas over H ({bH:.0f})", bH)]:
    r = clip_effect(yP["gas"], pP["gas"], np.full(len(P), cg), "gas"); line(tag, r); out[f"E2 {tag}"] = r
for tag, cg in [("eui clip 1e6/A", CAP), ("eui clip maxgasH/A", bH)]:
    r = clip_effect(yP["eui"], pP["eui"], cg / AP, "eui"); line(tag, r); out[f"E2 {tag}"] = r
g2 = out["E2 gas clip 1e6"]["dtotal"] + out["E2 eui clip 1e6/A"]["dtotal"]
print(f"  E2 gas+eui (1e6, 1e6/A) dTotal {g2:+.8f}")

print("\nE3  H re-split, bound from complement (P + other H parts)")
for name, k, seed in [("halves", 2, 20261020), ("5-fold", 5, 123)]:
    lab = np.random.default_rng(seed).permutation(len(H)) % k
    tot = []
    for f in range(k):
        s = lab == f
        bound = max(yP["gas"].max(), yH["gas"][~s].max())
        rg = clip_effect(yH["gas"][s], cH["gas"][s], np.full(s.sum(), bound), "gas")
        re = clip_effect(yH["eui"][s], cH["eui"][s], bound / AH[s], "eui")
        tot.append(rg["dtotal"] + re["dtotal"])
        print(f"  {name} part {f}: bound {bound:.0f}  gas rows {rg['n']:2d} dTotal {rg['dtotal']:+.8f}  "
              f"eui rows {re['n']:2d} dTotal {re['dtotal']:+.8f}  sum {tot[-1]:+.8f}")
    out[f"E3 {name}"] = tot
    print(f"  {name}: all parts negative-or-zero: {all(t <= 0 for t in tot)}; mean part dTotal {np.mean(tot):+.8f}")

print("\nE4  bootstrap of the H effect (gas 1e6 + eui 1e6/A)")
rng = np.random.default_rng(99)
qg, qe = np.minimum(cH["gas"], CAP), np.minimum(cH["eui"], CAP / AH)
eg0, eg1 = (yH["gas"] - cH["gas"]) ** 2, (yH["gas"] - qg) ** 2
ee0, ee1 = (yH["eui"] - cH["eui"]) ** 2, (yH["eui"] - qe) ** 2
bs = []
for _ in range(2000):
    i = rng.integers(0, len(H), len(H))
    dg = (np.sqrt(eg1[i].mean()) - np.sqrt(eg0[i].mean())) / SIG["gas"]
    de = (np.sqrt(ee1[i].mean()) - np.sqrt(ee0[i].mean())) / SIG["eui"]
    bs.append((dg + de) / 5)
bs = np.array(bs)
print(f"  dTotal bootstrap mean {bs.mean():+.8f} sd {bs.std():.8f}  share <= 0: {np.mean(bs <= 0):.3f}  "
      f"5%..95%: {np.quantile(bs, 0.05):+.8f} .. {np.quantile(bs, 0.95):+.8f}")
out["E4"] = dict(mean=float(bs.mean()), sd=float(bs.std()), share_le0=float(np.mean(bs <= 0)))

print("\nconditional truth for rows predicted above 1e6 (used for the test-side estimate)")
for tag, y, p in [("H corrected", yH["gas"], cH["gas"]), ("H teacher", yH["gas"], tH["gas"]), ("P teacher", yP["gas"], pP["gas"])]:
    s = p > CAP
    print(f"  {tag:<12s} rows {s.sum():3d}  mean pred {p[s].mean():10.0f}  mean y {y[s].mean():10.0f}  "
          f"min y {y[s].min():9.0f}  max y {y[s].max():9.0f}  mean excess {np.mean(p[s] - CAP):8.0f}")
    out[f"cond {tag}"] = dict(n=int(s.sum()), mean_y=float(y[s].mean()), mean_pred=float(p[s].mean()))
for tag, y, p, a in [("H corrected eui", yH["eui"], cH["eui"], AH), ("P teacher eui", yP["eui"], pP["eui"], AP)]:
    s = p * a > CAP
    print(f"  {tag:<16s} rows {s.sum():3d}  mean pred*A {np.mean(p[s] * a[s]):10.0f}  mean y*A {np.mean(y[s] * a[s]):10.0f}")
    out[f"cond {tag}"] = dict(n=int(s.sum()), mean_yA=float(np.mean(y[s] * a[s])))
json.dump(out, open("exp/audit/verify2_segments/hc_chain.json", "w"), indent=1)
print(f"\nmaxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
