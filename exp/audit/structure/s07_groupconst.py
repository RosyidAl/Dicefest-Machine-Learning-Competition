"""Are targets (or per-meter / per-area / per-EPC ratios) constant or exactly proportional within groups?
Groups: subzone, zone, climate cell, identical census vector, identical climate+zone. For each: share of multi-row groups
where the quantity is identical (rtol 1e-9) and the between-group share of log-variance (eta^2, leave-own-row unbiased not needed here)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
names = pq.read_schema("cache/train.parquet").names
socio = sorted(c for c in names if c.startswith("socio_demog") and c != "socio_demog_005")
T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
base = ["bld_dim_12", "bld_dim_01", "bld_dim_06", "bld_dim_03", "subzone_id", "zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"]
tr = pd.read_parquet("cache/train.parquet", columns=base + T)
soc = pd.read_parquet("cache/train.parquet", columns=socio)
# census constant within subzone?
nun = soc.join(tr.subzone_id).groupby("subzone_id").nunique()
const_share = (nun <= 1).mean()
print("socio columns constant within subzone: share of subzones per column, min/median/max", const_share.min(), const_share.median(), const_share.max())
print("columns not subzone-constant (<0.99):", list(const_share[const_share < 0.99].index)[:20])
groups = {
    "subzone": pd.factorize(tr.subzone_id)[0],
    "zone": pd.factorize(tr.zone_region_id)[0],
    "cell": pd.factorize(pd.util.hash_pandas_object(tr[["env_climate_01", "env_climate_03", "env_climate_05"]], index=False))[0],
    "census_vec": pd.factorize(pd.util.hash_pandas_object(soc, index=False))[0],
}
del soc
g, e, u = (tr[c].values for c in T); M = tr.bld_dim_12.values; A = tr.bld_dim_06.values; E = tr.bld_dim_01.values; G3 = tr.bld_dim_03.values
Q = {"gas": g, "gas/M": g / M, "eui": u, "gas/E": g / E, "gas/b03": g / G3, "elec": e, "elec/M": e / M, "elec/E": e / E, "gas/elec": g / e}
for gn, code in groups.items():
    ct = np.bincount(code); multi = ct[code] > 1
    print(f"[{gn}] groups {len(ct)}, multi-row groups {(ct > 1).sum()}, rows in multi {multi.mean():.3f}")
    for qn, q in Q.items():
        lq = np.log(q)
        df = pd.DataFrame({"c": code[multi], "q": q[multi], "lq": lq[multi]})
        s = df.groupby("c").q.agg(["min", "max"])
        ident = np.mean(np.isclose(s["min"], s["max"], rtol=1e-9, atol=0))
        # unbiased between-group share via leave-one-out mean
        sm = np.bincount(code, weights=lq); loo = (sm[code] - lq) / np.maximum(ct[code] - 1, 1)
        r2 = 1 - np.mean((lq[multi] - loo[multi]) ** 2) / np.var(lq[multi])
        print(f"   {qn:9s} identical-in-group share {ident:.4f}  LOO-mean R2(log) {r2:.3f}")
