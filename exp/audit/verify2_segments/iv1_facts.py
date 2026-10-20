"""Independent re-verification (lens verify2_segments, round 'iv'): raw facts behind the gas 1e6 truncation claim and an
independent rebuild of the candidate CSV with my own Mills-ratio implementation (erfcx form).

Reads only column subsets of cache/train.parquet / cache/test.parquet. Writes nothing except its own log (stdout).
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_segments/iv1_facts.py
"""
import hashlib
import resource
import time

import numpy as np
import pandas as pd
from scipy.special import erfcx

t0 = time.time()
GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
CL, AD = "target_clean_building_ratio", "target_electric_heating_adoption"
CAP = 1e6


def mills_shift(g, k, cap=CAP):
    """E[Y | Y < cap] - g for Y ~ N(g, (k g)^2): -s * phi(a) / Phi(a), with phi/Phi = sqrt(2/pi) / erfcx(-a / sqrt 2)."""
    s = k * g
    a = (cap - g) / s
    return -s * np.sqrt(2.0 / np.pi) / erfcx(-a / np.sqrt(2.0))


bld = [f"bld_dim_{i:02d}" for i in range(1, 19)]
tr = pd.read_parquet("cache/train.parquet", columns=["subzone_id", "socio_demog_005"] + bld + [GAS, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id", "subzone_id", "socio_demog_005"] + bld)
n_tr, n_te = len(tr), len(te)
print(f"train {n_tr} test {n_te} share {n_tr / (n_tr + n_te):.7f}  0.7*total = {0.7 * (n_tr + n_te):.3f}  "
      f"ceil(0.3*total) = {int(np.ceil(0.3 * (n_tr + n_te)))}")
g = tr[GAS].values
A = tr["bld_dim_06"].values
print(f"gas max {g.max():.4f}  min {g.min():.4f}  rows >= 1e6: {(g >= 1e6).sum()}  rows >= 999e3: {(g >= 999e3).sum()}")
edges = np.array([0.90, 0.95, 0.98, 0.99, 1.00, 1.01]) * 1e6
print("counts in [0.90,0.95,0.98,0.99,1.00,1.01]e6:", np.histogram(g, edges)[0].tolist())
print("eui*A vs gas: max rel diff %.3e" % np.max(np.abs(tr[EUI].values * A / g - 1)))
# tail shape: counts per 20k bin and a log-linear extrapolation fitted on [0.60, 0.90)e6
e = np.arange(0.60e6, 1.10e6 + 1, 0.02e6)
c = np.histogram(g, e)[0]
mid = 0.5 * (e[1:] + e[:-1])
fit = (mid < 0.90e6) & (c > 0)
b1, b0 = np.polyfit(mid[fit], np.log(c[fit]), 1, w=np.sqrt(c[fit]))
pred = np.exp(b0 + b1 * mid)
print("20k bins from 0.60e6: observed", c.tolist())
print("      log-linear fit on [0.60,0.90)e6, expected:", np.round(pred).astype(int).tolist())
print(f"  expected rows in [0.90,1.00)e6 {pred[(mid > 0.9e6) & (mid < 1e6)].sum():.0f} (obs {c[(mid > 0.9e6) & (mid < 1e6)].sum()})"
      f"  in [1.00,1.10)e6 {pred[mid > 1e6].sum():.0f} (obs {c[mid > 1e6].sum()})")

# submissions
base = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
cand_path = "exp/audit/segments/sub_v19vce_cap_k0.08.csv"
cand = pd.read_csv(cand_path, float_precision="round_trip")
assert (base["id"].values == te["id"].values).all() and (cand["id"].values == te["id"].values).all()
At = te["bld_dim_06"].values
for name, s in (("sub_v19vce", base), ("candidate", cand)):
    gp, ep = s[GAS].values, s[EUI].values
    print(f"{name}: gas > 1e6 rows {(gp > 1e6).sum()} max {gp.max():.1f}; eui*A > 1e6 rows {(ep * At > 1e6).sum()} "
          f"max eui*A {(ep * At).max():.1f}; gas > 0.95e6 {(gp > 0.95e6).sum()}  > 0.9e6 {(gp > 0.9e6).sum()}")

# duplicate test rows (my own key: subzone + socio_demog_005 + all 18 bld_dim columns)
key = ["subzone_id", "socio_demog_005"] + bld
ref = tr.groupby(key, sort=False)[[GAS, EUI]].agg(["first", "nunique"])
ref.columns = ["gas_first", "gas_nu", "eui_first", "eui_nu"]
ref = ref.reset_index()
m = te[key].merge(ref, on=key, how="left")
isdup = ~np.isnan(m["gas_first"].values)
print(f"duplicate test rows (18 bld dims + subzone + socio_005): {isdup.sum()}; consistent groups: "
      f"{bool((m.loc[isdup, 'gas_nu'] == 1).all())}")
print(f"  base gas on dup rows == train gas: {np.array_equal(base.loc[isdup, GAS].values, m.loc[isdup, 'gas_first'].values)}"
      f"; candidate dup cells == base (all 5 targets): "
      f"{np.array_equal(cand.loc[isdup].values[:, 1:].astype(float), base.loc[isdup].values[:, 1:].astype(float))}")

# independent rebuild with k = 0.08 on non-duplicate rows, gas > 6e5 mask, eui via eui*A
k = 0.08
mine = base.copy()
gb = base[GAS].values.astype(float)
mk = (gb > 6e5) & ~isdup
gnew = gb.copy()
gnew[mk] = gb[mk] + mills_shift(gb[mk], k)
eb = base[EUI].values.astype(float)
gE = eb * At
mkE = (gE > 6e5) & ~isdup
enew = eb.copy()
enew[mkE] = eb[mkE] * (gE[mkE] + mills_shift(gE[mkE], k)) / gE[mkE]
print(f"rebuild k={k}: gas rows changed {int((gnew != gb).sum())}  eui rows changed {int((enew != eb).sum())}")
print(f"  vs candidate: gas max |diff| {np.max(np.abs(gnew - cand[GAS].values)):.3e} kWh, "
      f"eui max |rel diff| {np.max(np.abs(enew / cand[EUI].values - 1)):.3e}")
print(f"  candidate - base: gas rows changed {(cand[GAS].values != gb).sum()}, > 0.1 sigma {(np.abs(cand[GAS].values - gb) > 0.1 * 151750.04).sum()}, "
      f"max change {np.max(gb - cand[GAS].values):.0f}, any increase {bool((cand[GAS].values > gb).any())}, new max {cand[GAS].max():.1f}")
for col in ["id", EL, CL, AD]:
    print(f"  column {col}: identical to base {bool((cand[col].values == base[col].values).all())}")
# rows changed on dup rows by my rebuild (should be zero by construction) and whether a dup row would have been hit
print(f"  dup rows with base gas > 6e5: {int((gb > 6e5)[isdup].sum())} (left untouched)")
print("candidate sha256", hashlib.sha256(open(cand_path, "rb").read()).hexdigest())
print(f"elapsed {time.time() - t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB")
