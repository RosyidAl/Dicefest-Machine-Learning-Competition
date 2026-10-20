"""Candidate: sub_v19vce + gas-cap truncation calibration (never submitted by this script).

gas' = g - s * phi(a) / Phi(a), s = k g, a = (1e6 - g) / s, k = 0.08 (fit on full-train OOF and on H; P-side teacher
fit 0.085), applied to the gas column and to the EUI column via g = eui * A (eui' = eui * g' / g). Only rows with
g > 6e5 are touched (below that the adjustment is < 1e-9 relative); the 77 duplicate rows keep their copied train
targets. All other cells stay byte-identical (checked).

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_make_candidate.py [k]
"""
import hashlib
import io
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

K = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
U, LO = 1e6, 6e5
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
TARGETS = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm",
           "target_clean_building_ratio", "target_electric_heating_adoption"]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]


def cap_calib(g, k=K, cap=U):
    s = k * g
    a = (cap - g) / s
    return g - s * np.exp(norm.logpdf(a) - norm.logcdf(a))


tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGETS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
raw = open("submissions/sub_v19vce.csv", "rb").read()
sub = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
ref = tr.groupby(KEY, sort=False)[TARGETS].first().reset_index()
hit = ~np.isnan(te[KEY].merge(ref, on=KEY, how="left")[TARGETS].values[:, 0])
At = te["bld_dim_06"].values
g0 = sub[GAS].values
m1 = (g0 > LO) & ~hit
g1 = g0.copy()
g1[m1] = cap_calib(g0[m1])
e0 = sub[EUI].values
gE = e0 * At
m2 = (gE > LO) & ~hit
e1 = e0.copy()
e1[m2] = e0[m2] * cap_calib(gE[m2]) / gE[m2]
out = sub.copy()
out[GAS], out[EUI] = g1, e1
path = f"exp/audit/segments/sub_v19vce_cap_k{K:g}.csv"
out.to_csv(path, index=False)
chk = pd.read_csv(path, float_precision="round_trip")
print(f"k {K}: gas rows changed {int((chk[GAS].values != g0).sum())} (max |d| {np.abs(g1 - g0).max():.6g}, new max {g1.max():.6g}); "
      f"eui rows changed {int((chk[EUI].values != e0).sum())} (max |d| {np.abs(e1 - e0).max():.6g}); dup rows {int(hit.sum())} untouched")
for c in ["id"] + TARGETS:
    print(f"  {c}: identical cells {int((chk[c].values == sub[c].values).sum())} / {len(sub)}")
print("  rms change / sigma: gas", np.sqrt(np.mean((g1 - g0) ** 2)) / 151750.04, " eui", np.sqrt(np.mean((e1 - e0) ** 2)) / 40.65)
print("  wrote", path, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
