"""Independent check of the candidate CSV exp/audit/segments/sub_v19vce_cap_k0.08.csv against submissions/sub_v19vce.csv,
and transfer check of the near-cap prediction distribution (test vs H corrected vs full-train VC OOF).

- recompute gas' / eui' with an own implementation (log_ndtr), k = 0.08, only rows g > 6e5, duplicate rows untouched
- text-level (byte) comparison of every untouched field
- expected test gain from the H per-bin mean dSSE, re-weighted to the test bin counts (bin-transfer estimate)

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/v02_candidate.py \
       > exp/audit/verify_segments/v02_candidate.log
"""
import hashlib

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

SG, SE = 151750.04, 40.65
U, K, LO = 1e6, 0.08, 6e5
LOG2PI = 0.5 * np.log(2 * np.pi)
TARGETS = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm",
           "target_clean_building_ratio", "target_electric_heating_adoption"]
GAS, EL, EUI = TARGETS[0], TARGETS[1], TARGETS[2]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
CAND = "exp/audit/segments/sub_v19vce_cap_k0.08.csv"
BASE = "submissions/sub_v19vce.csv"


def cal(g, k=K, cap=U):
    s = k * g
    a = (cap - g) / s
    return g - s * np.exp(-0.5 * a * a - LOG2PI - log_ndtr(a))


print("sha256 candidate", hashlib.sha256(open(CAND, "rb").read()).hexdigest())
print("sha256 base     ", hashlib.sha256(open(BASE, "rb").read()).hexdigest())
b = pd.read_csv(BASE, float_precision="round_trip")
c = pd.read_csv(CAND, float_precision="round_trip")
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGETS)
assert (b["id"].values == te["id"].values).all() and (c["id"].values == te["id"].values).all()
ref = tr.groupby(KEY, sort=False)[TARGETS].first().reset_index()
hit = ~np.isnan(te[KEY].merge(ref, on=KEY, how="left")[TARGETS].values[:, 0])
At = te["bld_dim_06"].values
g0, e0 = b[GAS].values, b[EUI].values
gE = e0 * At
print(f"dup rows {int(hit.sum())}; dup rows with gas > {LO:g}: {int((hit & (g0 > LO)).sum())}, with eui*A > {LO:g}: "
      f"{int((hit & (gE > LO)).sum())}")
m1 = (g0 > LO) & ~hit
m2 = (gE > LO) & ~hit
g1, e1 = g0.copy(), e0.copy()
g1[m1] = cal(g0[m1])
e1[m2] = e0[m2] * cal(gE[m2]) / gE[m2]
for name, mine, cand, base in [("gas", g1, c[GAS].values, g0), ("eui", e1, c[EUI].values, e0)]:
    rel = np.abs(mine - cand) / np.maximum(np.abs(cand), 1e-12)
    ch = cand != base
    print(f"{name}: own recompute vs candidate max rel diff {rel.max():.2e}; candidate cells changed {int(ch.sum())}, "
          f"all changed rows inside the mask: {bool(np.all(~ch | (m1 if name == 'gas' else m2)))}; "
          f"rows changed > 0.1 sigma {int(np.sum(np.abs(cand - base) > 0.1 * (SG if name == 'gas' else SE)))}; "
          f"max |change| {np.abs(cand - base).max():.6g}; max before {base.max():.6g} after {cand.max():.6g}; "
          f"min after {cand.min():.6g}; dup cells unchanged {bool(np.all(cand[hit] == base[hit]))}")
# monotonic mapping?
gg = np.linspace(6e5, 1.2e6, 20001)
print("mapping monotone on [6e5, 1.2e6]:", bool(np.all(np.diff(cal(gg)) > 0)), " cal(1e6) =", round(float(cal(1e6)), 1),
      " cal(1.2e6) =", round(float(cal(1.2e6)), 1), " cal(9e5) =", round(float(cal(9e5)), 1))

# byte-level comparison of the untouched fields
lb = open(BASE).read().splitlines()
lc = open(CAND).read().splitlines()
assert len(lb) == len(lc)
same = np.zeros(6, dtype=np.int64)
for x, y in zip(lb, lc):
    fx, fy = x.split(","), y.split(",")
    same += np.array([u == v for u, v in zip(fx, fy)], dtype=np.int64)
print("text fields identical per column (header + rows):", dict(zip(["id"] + TARGETS, same.tolist())), "of", len(lb))

# transfer: distribution of predictions near the cap
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
full = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
pcH, yH = z["gas_corrected"], z["gas_y"]
bins = [0.6, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 9]
print("\nshare of rows per bin of gas pred / 1e6 (per 100k rows): test sub_v19vce | H corrected | full VC OOF")
for lo, hi in zip(bins[:-1], bins[1:]):
    f = lambda p: 1e5 * np.mean((p / U >= lo) & (p / U < hi))  # noqa: E731
    print(f"   [{lo:.2f},{hi:.2f})  {f(g0):7.1f}  {f(pcH):7.1f}  {f(full['gas_p']):7.1f}")
# bin-transfer estimate of the test gas gain: H mean dSSE per row in each bin, weighted by test counts
pn = cal(np.where(pcH > LO, pcH, LO)) * (pcH > LO) + pcH * (pcH <= LO)
d = (yH - pn) ** 2 - (yH - pcH) ** 2
sse0 = np.mean((yH - pcH) ** 2)
edges = [LO, 8e5, 8.5e5, 9e5, 9.3e5, 9.6e5, 9.8e5, 1e6, 1.02e6, 1e7]
tot_t = 0.0
for lo, hi in zip(edges[:-1], edges[1:]):
    mH = (pcH >= lo) & (pcH < hi)
    nT = np.sum((g0 >= lo) & (g0 < hi) & ~hit)
    tot_t += nT * d[mH].mean() if mH.sum() else 0.0
est_T = (np.sqrt(sse0 + tot_t / len(g0)) - np.sqrt(sse0)) / SG / 5
est_H = (np.sqrt(sse0 + d.sum() / len(yH)) - np.sqrt(sse0)) / SG / 5
print(f"\ngas dTotal: on H {est_H:+.8f}; bin-transfer estimate for test {est_T:+.8f}")
