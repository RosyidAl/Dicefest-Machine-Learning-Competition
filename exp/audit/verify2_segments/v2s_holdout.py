"""verify2_segments, part B: honest size of an upper clip on gas (and EUI = gas / A) with splits other than P -> H.

Claim under test (segments lens): clipping gas at 1e6 (or at the train max) is worth about -0.000003 total
(H gas dNRMSE -0.0000158 on 11 rows, EUI clip at gasmax/A -0.0000004).

Independent evaluations here (own code, not seg_bounds.py):
  E1  constant cap 1e6 on all H rows (cap needs no labels: it is the data's truncation point).
  E2  split-half cross-fit on H, 25 seeds: cap = max gas label of half A, applied to half B and vice versa.
  E3  the other direction: cap = max gas label of H, applied to the P-side blend predictions (gas_pP, eui_pP),
      scored on the 300,512 P rows (labels y = pP + rP, checked against the train parquet).
  E4  paired bootstrap of the H delta (gas + EUI together), 2000 draws.
  E5  projection to the 7 / 8 test rows of sub_v19vce: expected SSE change uses E[y | prediction above the cap],
      estimated on H and on P rows only (never on test), then converted to a total-score change on 183,987 rows.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_segments/v2s_holdout.py
"""
import numpy as np
import pandas as pd

SG, SE = 151750.04, 40.65
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
CAP = 1e6
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH, AP = A[H], A[P]
gy, gc, ey, ec = z["gas_y"], z["gas_corrected"], z["eui_y"], z["eui_corrected"]
ytr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI])
assert np.allclose(ytr[GAS].values[H], gy, rtol=0, atol=1e-6) and np.allclose(ytr[EUI].values[H], ey, rtol=0, atol=1e-9)
gyP, eyP = z["gas_pP"] + z["gas_rP"], z["eui_pP"] + z["eui_rP"]
print("P-side labels = pP + rP: gas max abs diff %.3e  eui %.3e" % (np.abs(gyP - ytr[GAS].values[P]).max(),
                                                                  np.abs(eyP - ytr[EUI].values[P]).max()))


def nr(err, s):
    return np.sqrt(np.mean(err ** 2)) / s


def delta(y, p, cap, s):
    """NRMSE change from clipping p at cap (scalar or per-row), rows changed, rows where the clip made it worse"""
    q = np.minimum(p, cap)
    ch = q != p
    worse = int(np.sum(np.abs(q - y)[ch] > np.abs(p - y)[ch]))
    return nr(y - q, s) - nr(y - p, s), int(ch.sum()), worse


# ---------- E1: constant cap on all H ----------
print("\nE1 base NRMSE on H: gas %.6f  eui %.6f" % (nr(gy - gc, SG), nr(ey - ec, SE)))
dg, kg, wg = delta(gy, gc, CAP, SG)
de, ke, we = delta(ey, ec, CAP / AH, SE)
print(f"E1 H cap 1e6:     gas dNRMSE {dg:+.8f} ({kg} rows, {wg} worse)  eui(1e6/A) dNRMSE {de:+.8f} ({ke} rows, {we} worse)"
      f"  dTotal gas {dg / 5:+.9f} eui {de / 5:+.9f} both {(dg + de) / 5:+.9f}")
print("E1 H rows above cap: pred / truth:", " ".join(f"{p:.0f}/{t:.0f}" for p, t in sorted(zip(gc[gc > CAP], gy[gc > CAP]), reverse=True)))
print("E1 H eui rows with eui*A > 1e6: pred*A / truth*A:",
      " ".join(f"{p:.0f}/{t:.0f}" for p, t in sorted(zip((ec * AH)[ec * AH > CAP], (ey * AH)[ec * AH > CAP]), reverse=True)))

# ---------- E2: split-half cross-fit on H, cap learned from the other half's labels ----------
rows = []
for seed in range(25):
    half = np.random.default_rng(1000 + seed).random(len(H)) < 0.5
    qg, qe = gc.copy(), ec.copy()
    for fit, ev in ((half, ~half), (~half, half)):
        cap = gy[fit].max()
        qg[ev] = np.minimum(gc[ev], cap)
        qe[ev] = np.minimum(ec[ev], cap / AH[ev])
    d_g = nr(gy - qg, SG) - nr(gy - gc, SG)
    d_e = nr(ey - qe, SE) - nr(ey - ec, SE)
    # per-half deltas (each half scored on its own rows, cap from the other half)
    hd = [nr(gy[ev] - qg[ev], SG) - nr(gy[ev] - gc[ev], SG) for ev in (half, ~half)]
    rows.append((seed, d_g, d_e, (d_g + d_e) / 5, hd[0] / 5, hd[1] / 5))
r = np.array(rows)
print(f"\nE2 split-half cross-fit (25 seeds): dTotal gas mean {r[:, 1].mean() / 5:+.9f}  eui mean {r[:, 2].mean() / 5:+.9f}"
      f"  both mean {r[:, 3].mean():+.9f} (min {r[:, 3].min():+.9f} max {r[:, 3].max():+.9f})")
print(f"E2 per-half gas dTotal: min {r[:, 4:].min():+.9f}  max {r[:, 4:].max():+.9f}  share of halves < 0: {(r[:, 4:] < 0).mean():.2f}"
      f"  (= 0 means the half had no row above the cap)")

# ---------- E3: cap from H labels, applied to P-side predictions, scored on P ----------
capH = gy.max()
pgP, peP = z["gas_pP"], z["eui_pP"]
print("\nE3 base NRMSE on P (P-side blend, uncorrected): gas %.6f  eui %.6f" % (nr(gyP - pgP, SG), nr(eyP - peP, SE)))
for name, cap in (("cap max_H %.1f" % capH, capH), ("cap 1e6", CAP)):
    dg, kg, wg = delta(gyP, pgP, cap, SG)
    de, ke, we = delta(eyP, peP, cap / AP, SE)
    print(f"E3 P {name:18s}: gas dNRMSE {dg:+.8f} ({kg} rows, {wg} worse)  eui dNRMSE {de:+.8f} ({ke} rows, {we} worse)"
          f"  dTotal both {(dg + de) / 5:+.9f}")
print("E3 P rows above 1e6: rate %.2e per row (H corrected: %.2e)" % (np.mean(pgP > CAP), np.mean(gc > CAP)))

# ---------- E4: paired bootstrap of the H delta ----------
rng = np.random.default_rng(7)
qg, qe = np.minimum(gc, CAP), np.minimum(ec, CAP / AH)
sg0, sg1, se0, se1 = (gy - gc) ** 2, (gy - qg) ** 2, (ey - ec) ** 2, (ey - qe) ** 2
bs = []
for _ in range(2000):
    i = rng.integers(0, len(H), len(H))
    bs.append(((np.sqrt(sg1[i].mean()) - np.sqrt(sg0[i].mean())) / SG + (np.sqrt(se1[i].mean()) - np.sqrt(se0[i].mean())) / SE) / 5)
bs = np.array(bs)
print(f"\nE4 bootstrap dTotal (gas+eui, cap 1e6) on H: mean {bs.mean():+.9f}  sd {bs.std():.9f}  "
      f"5%..95% {np.quantile(bs, 0.05):+.9f}..{np.quantile(bs, 0.95):+.9f}  share >= 0: {(bs >= 0).mean():.3f}")

# ---------- E5: projection onto the test rows of sub_v19vce ----------
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=[GAS, EUI], float_precision="round_trip")
At = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06"])["bld_dim_06"].values.astype(float)
pg, pe = sub[GAS].values, sub[EUI].values
n = len(pg)
# E[y | prediction above the cap] from labelled rows only: H corrected rows, and P-side rows, as two estimates
Ey = {"H": gy[gc > CAP].mean(), "P": gyP[pgP > CAP].mean(), "P_1.00-1.07e6": gyP[(pgP > CAP) & (pgP < 1.07e6)].mean()}
print("\nE5 E[gas | pred > 1e6]: " + "  ".join(f"{k} {v:.0f}" for k, v in Ey.items()))
base_g, base_e = nr(gy - gc, SG), nr(ey - ec, SE)  # H corrected NRMSE as the test base level
og, oe = pg > CAP, pe * At > CAP
for k, m in Ey.items():
    dsse_g = np.sum((pg[og] - CAP) * (pg[og] + CAP - 2 * m))
    ce = CAP / At[oe]
    dsse_e = np.sum((pe[oe] - ce) * (pe[oe] + ce - 2 * m / At[oe]))  # E[eui] = E[gas] / A
    dg = np.sqrt(base_g ** 2 - dsse_g / n / SG ** 2) - base_g
    de = np.sqrt(base_e ** 2 - dsse_e / n / SE ** 2) - base_e
    print(f"E5 test projection with E[y]={k:14s}: gas rows {og.sum()} dTotal {dg / 5:+.9f}  eui rows {oe.sum()} dTotal {de / 5:+.9f}"
          f"  both {(dg + de) / 5:+.9f}")
# worst case for the clip: truth exactly at the cap (smallest possible gain) and best case: truth at the H minimum seen
for k, m in (("truth=cap (min gain)", CAP), ("truth=850k", 850e3)):
    dsse_g = np.sum((pg[og] - CAP) * (pg[og] + CAP - 2 * m))
    ce = CAP / At[oe]
    dsse_e = np.sum((pe[oe] - ce) * (pe[oe] + ce - 2 * m / At[oe]))
    dg = np.sqrt(base_g ** 2 - dsse_g / n / SG ** 2) - base_g
    de = np.sqrt(base_e ** 2 - dsse_e / n / SE ** 2) - base_e
    print(f"E5 test bound {k:22s}: dTotal gas {dg / 5:+.9f}  eui {de / 5:+.9f}  both {(dg + de) / 5:+.9f}")
