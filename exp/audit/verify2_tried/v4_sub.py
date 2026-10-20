"""Test-side check (verify2_tried) of fix G on the real submission submissions/sub_v19vce.csv.

* which test rows the fix touches (gas > 9e5; implied gas eui*A > 9e5), excluding the 77 duplicate-copied rows
* size of the change (rms / sigma) for beta from the full-OOF VC teacher (v3) and from the P side (-0.7275)
* whether any duplicate-copied row lies above the knot (order of fix vs duplicate copy)
* consistency of the auditor's probe exp/audit/tried/sub_v19vce_GS_audit.csv with G on rows that S does not touch
Writes nothing but its log; never touches submissions/.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "final/src")
from common import TARGET_COLS  # noqa: E402
from dup_fix import KEY, duplicate_targets  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
SIG = {GAS: 151750.04, EUI: 40.65}
K, CAP = 9e5, 1e6
sub = pd.read_csv("submissions/sub_v19vce.csv")
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
assert (sub["id"].values == te["id"].values).all()
At = te["bld_dim_06"].values
Mt = te["bld_dim_12"].values.astype(float)
dup = duplicate_targets(tr, te)
hit = ~np.isnan(dup[:, 0])
print(f"duplicate rows {hit.sum()}; dup rows with copied gas > 9e5: {np.sum(dup[hit, 0] > K)}; "
      f"max copied gas {np.nanmax(dup[:, 0]):.0f}")
assert np.allclose(sub.loc[hit, GAS].values, dup[hit, 0])
g, e = sub[GAS].values, sub[EUI].values
print(f"sub gas > 9e5: {np.sum(g > K)} (non-dup {np.sum((g > K) & ~hit)}), > 1e6: {np.sum(g > CAP)}, max {g.max():.0f}; "
      f"implied eui*A > 9e5: {np.sum(e * At > K)}, > 1e6: {np.sum(e * At > CAP)}")
pf_beta = None
try:
    pfg = np.load("exp/audit/verify2_tried/full_pf_gas.npy")
    yg = tr[GAS].values
    h = np.maximum(pfg - K, 0)
    pf_beta = float(h @ (yg - pfg) / (h @ h))
except FileNotFoundError:
    pass


def apply(g, e, beta):
    g2 = np.where((g > K) & ~hit, np.minimum(g + beta * (g - K), CAP), g)
    ge = e * At
    e2 = np.where((ge > K) & ~hit, np.minimum(ge + beta * (ge - K), CAP) / At, e)
    return g2, e2


for tag, beta in (("full-OOF VC teacher", pf_beta), ("P side", -0.7274832632204717)):
    if beta is None:
        continue
    g2, e2 = apply(g, e, beta)
    dg, de = g2 - g, e2 - e
    print(f"beta {beta:+.4f} ({tag}): gas rows changed {np.sum(dg != 0)}, rms change/sigma "
          f"{np.sqrt(np.mean(dg ** 2)) / SIG[GAS]:.6f}, max |change| {np.abs(dg).max():.0f} kWh, new max {g2.max():.0f}; "
          f"eui rows changed {np.sum(de != 0)}, rms change/sigma {np.sqrt(np.mean(de ** 2)) / SIG[EUI]:.6f}")
    print(f"    changed rows M quantiles {np.quantile(Mt[dg != 0], [0, .5, 1]).tolist()}; "
          f"all changes negative: gas {bool(np.all(dg <= 0))} eui {bool(np.all(de <= 0))}")
# the auditor's G+S probe: on rows with M >= 5 (S untouched) gas/eui should equal G(sub) with the P-side beta
gs = pd.read_csv("exp/audit/tried/sub_v19vce_GS_audit.csv")
assert (gs["id"].values == sub["id"].values).all()
g2, e2 = apply(g, e, -0.7274832632204717)
big = Mt >= 5
for col, ref in ((GAS, g2), (EUI, e2)):
    d = np.abs(gs[col].values[big] - ref[big])
    print(f"probe vs G(sub) on M>=5 rows, {col}: max abs diff {d.max():.3e}, rows differing (>1e-6 rel) "
          f"{np.sum(d > 1e-6 * np.abs(ref[big]))}")
for col in (EL, TARGET_COLS[3], TARGET_COLS[4]):
    print(f"probe vs sub {col}: rows differing {np.sum(gs[col].values != sub[col].values)} "
          f"(M>=5 rows differing {np.sum(gs[col].values[big] != sub[col].values[big])})")
# is the TEST truncated too? Compare the count of high test predictions with the rate on H (a truncated random
# subset of train, predicted the same way: fold-averaged members + correction). An untruncated test would contain
# ~0.3/0.7 * (train rows missing above 1e6) extra rows whose predictions sit mostly above 8e5.
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
cg = z["gas_corrected"]
nH, nT = len(cg), len(g)
for thr in (7e5, 8e5, 9e5, 9.5e5, 1e6):
    exp_cnt = np.sum(cg > thr) * nT / nH
    obs = np.sum(g[~hit] > thr) + np.sum(g[hit] > thr)
    print(f"pred > {thr:9.0f}: H count {np.sum(cg > thr):4d} -> expected on test {exp_cnt:7.1f} (Poisson sd "
          f"{np.sqrt(exp_cnt):5.1f}); observed on test {obs}")
MH = z["M"][z["H"]]
for lo in (60, 80, 100):
    print(f"M >= {lo}: H {np.sum(MH >= lo)} -> expected test {np.sum(MH >= lo) * nT / nH:7.1f}; observed {np.sum(Mt >= lo)}")
