"""Adversarial re-derivation of the claim: '7 test gas predictions in sub_v19vce exceed the hard data maximum of 1e6;
clipping gas at 1e6 (and EUI at 1e6/A) is worth about -0.000003 total'.

Checks
 1. Is the gas target really hard-truncated below 1e6 (train tail density, identity eui == gas/A)?
 2. Does the truncation apply to the test rows as well (random 70/30 split)? A train-only filter would leave the test
    with an excess of very large rows: compare train/test counts in the tail of floor area A, meters M and of a model
    proxy prediction (equal-weight mean of the gas-family members, OOF on train / fold-averaged on test).
 3. Honest H effect: fixed bounds (1e6, or max of P targets only) applied to the holdout-chain corrected predictions.
    No parameter is fitted on H. Paired bootstrap of the delta over H rows.
 4. Test-side effect: the actual 7 / 8 affected rows of the submitted file; lower bound (truth at the cap) and expected
    change (truth gap taken from H rows that were predicted above the cap). Converted to total score.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/v1_clip_verify.py \
          > exp/audit/verify_segments/v1_clip_verify.log
"""
import json

import numpy as np
import pandas as pd
import yaml

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
CAP = 1e6
out = {}

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12"])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["id", GAS, EUI])
assert np.array_equal(tr["bld_dim_06"].values.astype(float), A), "A in chain_H differs from train bld_dim_06"
assert np.array_equal(te["id"].values, sub["id"].values), "test id order differs from submission"
ntr, nte = len(tr), len(te)
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
yg, ye = tr[GAS].values, tr[EUI].values
assert np.allclose(z["gas_y"], yg[H]) and np.allclose(z["eui_y"], ye[H])


def nr(e, m):
    return float(np.sqrt(np.mean(e ** 2)) / SIG[m])


# ---------------------------------------------------------------- 1. truncation evidence (train)
print("=== 1. truncation evidence")
print(f"train rows {ntr}, test rows {nte}, ratio {ntr / nte:.4f}")
print(f"gas max {yg.max():.1f}, rows >= 1e6: {(yg >= CAP).sum()}, rows > 999765: {(yg > 999765).sum()}")
rel = np.abs(ye * A - yg) / yg
print(f"identity eui*A == gas: max rel err {rel.max():.2e}")
edges = np.arange(700_000, 1_000_001, 25_000)
cnt, _ = np.histogram(yg, bins=edges)
print("gas counts per 25k bin from 700k to 1e6:", " ".join(str(c) for c in cnt))
# exponential tail fit on [700k, 1e6): log count ~ a + b * bin_mid, extrapolate the mass above 1e6
mid = 0.5 * (edges[:-1] + edges[1:])
b, a = np.polyfit(mid, np.log(cnt), 1)
scale = -1.0 / b  # kWh per e-fold
dens_at_cap = np.exp(a + b * CAP) / 25_000  # rows per kWh at 1e6
expected_above = dens_at_cap * scale  # integral of the exponential tail above 1e6
print(f"exp tail fit: e-fold {scale:.0f} kWh; density at 1e6 {dens_at_cap * 1e4:.2f} rows/10k kWh;"
      f" expected train rows above 1e6 without truncation ~{expected_above:.0f} (observed 0)")
out["trunc"] = dict(gas_max=float(yg.max()), n_ge_cap=int((yg >= CAP).sum()), tail_counts_25k=cnt.tolist(),
                    expected_above_if_untruncated=float(expected_above), identity_max_rel_err=float(rel.max()))

# ---------------------------------------------------------------- 2. is the test truncated too?
print("\n=== 2. train-only-filter check (train/test count ratio in tails; null ratio %.3f)" % (ntr / nte))
members = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
oof = np.mean([np.load(f"exp/oof/{r}.npy") for r in members], axis=0) * A  # EUI scale -> gas
tpr = np.mean([np.load(f"exp/pred/{r}.npy") for r in members], axis=0) * At
assert oof.shape == (ntr,) and tpr.shape == (nte,)
rows = []
for name, xtr, xte in [("A", A, At), ("M", M, Mt), ("proxy_pred_gas", oof, tpr)]:
    for thr in ([5e5, 7e5, 8e5, 9e5, 9.5e5, 1e6, 1.05e6] if name == "proxy_pred_gas"
                else list(np.quantile(np.r_[xtr, xte], [0.99, 0.995, 0.999, 0.9995, 0.9999]))):
        a_, b_ = int((xtr > thr).sum()), int((xte > thr).sum())
        # binomial z-score of the test share under the null share nte/(ntr+nte)
        p0 = nte / (ntr + nte)
        n = a_ + b_
        zsc = (b_ - n * p0) / np.sqrt(n * p0 * (1 - p0)) if n else 0.0
        rows.append((name, thr, a_, b_, a_ / max(b_, 1), zsc))
        print(f"  {name:15s} > {thr:12.6g}: train {a_:6d} test {b_:6d} ratio {a_ / max(b_, 1):6.3f}  z(test excess) {zsc:+.2f}")
out["filter_check"] = [dict(feature=r[0], thr=float(r[1]), n_train=r[2], n_test=r[3], ratio=r[4], z=float(r[5]))
                       for r in rows]
# under the alternative the test would hold ~expected_above * nte/ntr extra rows with true gas >= 1e6
print(f"  alt hypothesis (train-only filter): test would hold ~{expected_above * nte / ntr:.0f} rows with gas >= 1e6,"
      f" nearly all with proxy pred > 7e5")

# ---------------------------------------------------------------- 3. honest H effect (fixed bounds, nothing fitted on H)
print("\n=== 3. holdout chain (H corrected = honest analog of the submission)")
B, CH = 2000, 50  # bootstrap replicates, processed in chunks to keep memory small


def h_effect(m, pc, pn, yH):
    base, new = nr(yH - pc, m), nr(yH - pn, m)
    ch = pn != pc
    imp = np.abs(yH - pn)[ch] < np.abs(yH - pc)[ch]
    # paired bootstrap of the NRMSE delta (resample H rows); squared errors precomputed
    s0, s1 = (yH - pc) ** 2, (yH - pn) ** 2
    rng = np.random.default_rng(0)  # same resamples for every candidate
    d = []
    for _ in range(B // CH):
        idx = rng.integers(0, len(yH), size=(CH, len(yH)))
        d.append(np.sqrt(s1[idx].mean(1)) / SIG[m] - np.sqrt(s0[idx].mean(1)) / SIG[m])
    d = np.concatenate(d)
    return base, new - base, int(ch.sum()), int(imp.sum()), float(d.std())


res = {}
gH, gc = z["gas_y"], z["gas_corrected"]
eH, ec = z["eui_y"], z["eui_corrected"]
AH = A[H]
maxP = yg[P].max()
for tag, m, pc, yH, hi in [("gas cap 1e6", "gas", gc, gH, CAP), ("gas cap max_P", "gas", gc, gH, maxP),
                           ("eui cap 1e6/A", "eui", ec, eH, CAP / AH), ("eui cap max_P/A", "eui", ec, eH, maxP / AH)]:
    pn = np.minimum(pc, hi)
    base, dn, k, kimp, sd = h_effect(m, pc, pn, yH)
    res[tag] = dict(base=base, dNRMSE=dn, dTotal=dn / 5, rows=k, rows_improved=kimp, boot_sd_dTotal=sd / 5)
    print(f"  {tag:16s}: base {base:.5f} rows {k:3d} improved {kimp:3d}  dNRMSE {dn:+.7f}  dTotal {dn / 5:+.8f}"
          f"  (boot sd {sd / 5:.8f})")
tot_h = res["gas cap 1e6"]["dTotal"] + res["eui cap 1e6/A"]["dTotal"]
print(f"  H total (gas 1e6 + eui 1e6/A): {tot_h:+.8f}")
# the same on the teacher (uncorrected) to show the correction does not create the effect
for m, pt, yH, hi in [("gas", z["gas_teacher"], gH, CAP), ("eui", z["eui_teacher"], eH, CAP / AH)]:
    pn = np.minimum(pt, hi)
    print(f"  teacher {m}: rows {(pn != pt).sum()}  dTotal {(nr(yH - pn, m) - nr(yH - pt, m)) / 5:+.8f}")
out["H"] = res
out["H_total"] = tot_h
sel = gc > CAP
gap_H = CAP - gH[sel]
print(f"  H rows with gas pred > 1e6: {sel.sum()}; truth gap (1e6 - y): mean {gap_H.mean():.0f} median "
      f"{np.median(gap_H):.0f} min {gap_H.min():.0f}")
near = (gc > 0.97e6)
print(f"  H rows with gas pred > 0.97e6: {near.sum()}; mean gap {np.mean(CAP - gH[near]):.0f}")

# ---------------------------------------------------------------- 4. test side
print("\n=== 4. test side (submitted file sub_v19vce.csv)")
pg, pe = sub[GAS].values, sub[EUI].values
kg = pg > CAP
ke = pe * At > CAP
print(f"  gas rows > 1e6: {kg.sum()}  values: " + " ".join(f"{v:.0f}" for v in np.sort(pg[kg])[::-1]))
print(f"  eui rows with eui*A > 1e6: {ke.sum()}  overlap with gas rows: {(kg & ke).sum()}")
print(f"  eui rows excess (EUI units): " + " ".join(f"{v:.3f}" for v in np.sort((pe - CAP / At)[ke])[::-1]))
print(f"  test gas rows > max_train ({yg.max():.0f}): {(pg > yg.max()).sum()}")
# expected change of SSE: dSE_i = -(e_i^2 + 2 e_i g_i), g_i = cap - y_i >= 0 (truth below the cap)
eg = pg[kg] - CAP
ee = (pe - CAP / At)[ke]
gap_e_scale = CAP / At[ke]  # 1 kWh of gas gap = 1/A EUI units
nrm_base = {"gas": res["gas cap 1e6"]["base"], "eui": res["eui cap 1e6/A"]["base"]}


def d_total(dsse, m, n):
    mse0 = (nrm_base[m] * SIG[m]) ** 2
    return (np.sqrt(mse0 + dsse / n) / SIG[m] - nrm_base[m]) / 5


scen = {"lower bound (truth at cap)": 0.0, "H median gap": float(np.median(gap_H)), "H mean gap": float(gap_H.mean())}
out["test"] = dict(gas_rows=int(kg.sum()), eui_rows=int(ke.sum()), gas_excess=eg.tolist(), eui_excess=ee.tolist())
for sname, g in scen.items():
    dg = -np.sum(eg ** 2 + 2 * eg * g)
    de = -np.sum(ee ** 2 + 2 * ee * g / At[ke])
    tg, tE = d_total(dg, "gas", nte), d_total(de, "eui", nte)
    print(f"  {sname:28s}: gas dTotal {tg:+.8f}  eui dTotal {tE:+.8f}  sum {tg + tE:+.8f}  (full test)")
    out["test"][sname] = dict(gap=g, gas_dTotal=float(tg), eui_dTotal=float(tE))
# private LB = 70% of test rows: the 7/8 rows land there at random; expected delta is about the full-test value
# (numerator and denominator both scale by 0.7), but realised value varies with how many rows land in private
json.dump(out, open("exp/audit/verify_segments/v1_clip_verify.json", "w"), indent=1, default=float)
print("\nwrote exp/audit/verify_segments/v1_clip_verify.json")
