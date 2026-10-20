"""verify2_segments: test-side check of the candidate exp/audit/segments/sub_v19vce_cap_k0.08.csv.

1. my own duplicate-row match (key socio_demog_005, subzone_id, bld_dim_12, bld_dim_01, bld_dim_06) on column subsets
2. my log_ndtr calibration (k 0.08) applied to sub_v19vce gas and EUI (via eui*A) except duplicate rows,
   compared numerically with the candidate
3. the ACTION's exact formula inserted in pipeline order (calibrate rows > 6e5, then copy duplicates), written to my
   folder and compared byte for byte with the candidate
4. level distribution of test gas predictions vs the H corrected chain
"""
import hashlib

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import norm

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
TC = [GAS, EL, EUI, "target_clean_building_ratio", "target_electric_heating_adoption"]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
CAP, K = 1e6, 0.08
base = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
cand = pd.read_csv("exp/audit/segments/sub_v19vce_cap_k0.08.csv", float_precision="round_trip")
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
trk = pd.read_parquet("cache/train.parquet", columns=KEY + TC)
assert (te["id"].values == base["id"].values).all() and (cand["id"].values == base["id"].values).all()
At = te["bld_dim_06"].values.astype(float)

# 1. duplicates
ref = trk.groupby(KEY, sort=False)[TC].agg(["min", "max"])
assert all((ref[(c, "min")] == ref[(c, "max")]).all() for c in TC)
first = trk.groupby(KEY, sort=False)[TC].first().reset_index()
mg = te[KEY].merge(first, on=KEY, how="left")
dup = ~mg[GAS].isna().values
print(f"duplicate test rows {dup.sum()}; base gas equals train value on all of them: "
      f"{np.allclose(base.loc[dup, GAS].values, mg.loc[dup, GAS].values, rtol=0, atol=1e-6)}")
print(f"dup rows with base gas > 6e5: {(base.loc[dup, GAS].values > 6e5).sum()}")


# 2. my calibration
def mills(a):
    return np.exp(-0.5 * a * a - 0.5 * np.log(2.0 * np.pi) - log_ndtr(a))


def calib(g, k=K):
    out = np.array(g, dtype=float, copy=True)
    m = out > 5e5
    s = k * out[m]
    out[m] = out[m] - s * mills((CAP - out[m]) / s)
    return out


g0, e0 = base[GAS].values, base[EUI].values
g1 = np.where(dup, g0, calib(g0))
gE = e0 * At
e1 = np.where(dup, e0, e0 * calib(gE) / gE)
dg = np.abs(g1 - cand[GAS].values)
de = np.abs(e1 - cand[EUI].values) / np.maximum(e0, 1e-9)
print(f"my calibration vs candidate: gas max |diff| {dg.max():.3e} kWh, eui max rel diff {de.max():.3e}")
for c in (EL, "target_clean_building_ratio", "target_electric_heating_adoption", "id"):
    print(f"  column {c}: identical to base {(cand[c].values == base[c].values).all()}")
print(f"  dup rows identical to base (all targets): {(cand.loc[dup, TC].values == base.loc[dup, TC].values).all()}")
chg = np.abs(cand[GAS].values - g0)
print(f"  gas: rows changed by > 0.1 sigma {(chg > 0.1 * 151750.04).sum()}, max change {chg.max():.0f},"
      f" new max {cand[GAS].max():.0f} (base max {g0.max():.0f}, base rows > 1e6: {(g0 > CAP).sum()})")
print(f"  eui*A: new max {(cand[EUI].values * At).max():.0f}; rows where the candidate is lower than base: gas "
      f"{(cand[GAS].values < g0).sum()} higher {(cand[GAS].values > g0).sum()}")


# 3. ACTION formula in pipeline order, byte comparison
def cap_calib(g, k=0.08, cap=1e6):
    s = k * g
    a = (cap - g) / s
    return g - s * np.exp(norm.logpdf(a) - norm.logcdf(a))


sub = base.copy()
m = sub[GAS].values > 6e5
sub.loc[m, GAS] = cap_calib(sub.loc[m, GAS].values)
gE = sub[EUI].values * At
m = gE > 6e5
sub.loc[m, EUI] = sub.loc[m, EUI].values * cap_calib(gE[m]) / gE[m]
for c in TC:  # step 3 of _assemble: duplicates get the train targets
    sub.loc[dup, c] = mg.loc[dup, c].values
out = "exp/audit/verify2_segments/sub_v19vce_cap_k0.08_verify2.csv"
sub.to_csv(out, index=False)
h_mine = hashlib.sha256(open(out, "rb").read()).hexdigest()
h_cand = hashlib.sha256(open("exp/audit/segments/sub_v19vce_cap_k0.08.csv", "rb").read()).hexdigest()
print(f"pipeline-order rebuild sha256 {h_mine}\ncandidate sha256             {h_cand}\nbyte-identical: {h_mine == h_cand}")
# does the base itself survive a round trip (dup rows written from train values are the same bytes)?
chk = base.copy()
for c in TC:
    chk.loc[dup, c] = mg.loc[dup, c].values
print(f"base dup cells equal train values exactly: {(chk[TC].values == base[TC].values).all()}")

# 4. level distribution vs H chain
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
gH = z["gas_corrected"]
for t in (0.85e6, 0.9e6, 0.95e6, 1e6):
    print(f"  share gas pred > {t:.2e}: test {np.mean(g0[~dup] > t):.5f} ({(g0[~dup] > t).sum()})"
          f"  H corrected {np.mean(gH > t):.5f} ({(gH > t).sum()})")
eH = z["eui_corrected"] * z["A"][z["H"]]
for t in (0.85e6, 0.9e6, 1e6):
    print(f"  share eui*A > {t:.2e}: test {np.mean(e0[~dup] * At[~dup] > t):.5f}  H corrected {np.mean(eH > t):.5f}")
