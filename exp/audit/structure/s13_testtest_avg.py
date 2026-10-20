"""Test-test exact duplicates (K0 key, no train partner): targets are identical in all 92 train K0 groups, so averaging the
predictions inside a test-only group can only lower the squared error: SSE drop per pair = (p1-p2)^2/2 exactly (for any y).
Also verify on H (pairs with both rows in H) and in train overall (all groups identical targets)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=K0); te = pd.read_parquet("cache/test.parquet", columns=K0)
ktr = pd.util.hash_pandas_object(tr[K0], index=False).values; kte = pd.util.hash_pandas_object(te[K0], index=False).values
sb = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
hit = np.isin(kte, ktr); st = pd.Series(kte); tm = (st.map(st.value_counts()).values > 1) & ~hit
codes = pd.factorize(kte)[0]
H_rmse = {"target_macro_gas_kwh": 0.24514, "target_macro_elec_kwh": 0.31462, "target_eui_gas_kwh_per_sqm": 0.64859}
N = len(te); tot = 0
new = sb.copy()
for c in TARGET_COLS[:3]:
    p = sb[c].values
    s = np.bincount(codes[tm], weights=p[tm], minlength=codes.max() + 1); n = np.bincount(codes[tm], minlength=codes.max() + 1)
    avg = p.copy(); avg[tm] = s[codes[tm]] / n[codes[tm]]
    dsse = ((p - avg) ** 2)[tm].sum()  # exact SSE drop when targets inside a group are identical
    mse = (H_rmse[c] * SIGMAS[c]) ** 2
    dn = np.sqrt(mse - dsse / N) / SIGMAS[c] - H_rmse[c]
    tot += dn / 5
    print(f"{c}: test-only dup rows {int(tm.sum())}, SSE drop {dsse:.4g}, NRMSE change ~{dn:+.7f}")
print(f"total change ~{tot:+.7f}  (public/private both: rows split at random)")
# H check: H rows whose K0 partners are in H too
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]
kH = ktr[H]; sH = pd.Series(kH); mH = sH.map(sH.value_counts()).values > 1
cH = pd.factorize(kH)[0]
for m in ["gas", "elec", "eui"]:
    y = d[f"{m}_y"]; p = d[f"{m}_corrected"]
    s = np.bincount(cH, weights=p); n = np.bincount(cH); avg = np.where(mH, s[cH] / n[cH], p)
    print(f"H {m}: rows in H-H dup groups {int(mH.sum())}, y identical inside groups: {bool(np.all(pd.Series(y[mH]).groupby(cH[mH]).nunique() == 1))}, "
          f"SSE before {((y - p)[mH] ** 2).sum():.4g} after {((y - avg)[mH] ** 2).sum():.4g}")
