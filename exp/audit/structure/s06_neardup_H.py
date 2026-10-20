"""Honest evaluation of relaxed near-duplicate copy rules on the holdout H.
For each key: H rows with >=1 OTHER train row sharing the key (partner labels are known at test time as well);
copy = mean partner target (or per-meter / per-area proportional copy). Compare with the honest corrected prediction.
Rows that match under the exact dup_fix key K0 are reported separately (already used in the submission)."""
import numpy as np, pandas as pd
S = {"gas": 151750.04, "elec": 41725.37, "eui": 40.65}
T = {"gas": "target_macro_gas_kwh", "elec": "target_macro_elec_kwh", "eui": "target_eui_gas_kwh_per_sqm"}
B = [f"bld_dim_{i:02d}" for i in range(1, 44)]
import pyarrow.parquet as pq
names = pq.read_schema("cache/train.parquet").names; B = [b for b in B if b in names]
cols = sorted(set(B + ["socio_demog_005", "subzone_id", "zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"]))
tr = pd.read_parquet("cache/train.parquet", columns=cols + list(T.values()))
te = pd.read_parquet("cache/test.parquet", columns=cols)
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
keys = {
    "K1 sub+M+E+A": ["subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"],
    "K2 allbld": B,
    "K3 sub+M+E": ["subzone_id", "bld_dim_12", "bld_dim_01"],
    "K4 sub+A": ["subzone_id", "bld_dim_06"],
    "K5 sub+M+E+02+03": ["subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_02", "bld_dim_03"],
    "K6 sub+M+E+17": ["subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_17"],
    "K7 M+E+A+03": ["bld_dim_12", "bld_dim_01", "bld_dim_06", "bld_dim_03"],
    "K8 A+03+clim": ["bld_dim_06", "bld_dim_03", "env_climate_01", "env_climate_03"],
    "K9 sub+M": ["subzone_id", "bld_dim_12"],
}
M = tr.bld_dim_12.values; A = tr.bld_dim_06.values
def gid(df, key):
    return pd.util.hash_pandas_object(df[key], index=False).values
k0tr = gid(tr, K0); k0cnt = pd.Series(k0tr).map(pd.Series(k0tr).value_counts()).values
inK0 = k0cnt[H] > 1
out = []
for name, key in keys.items():
    g = pd.Series(gid(tr, key)); n = g.map(g.value_counts()).values
    gte = pd.Series(gid(te, key)); test_match = int(gte.isin(set(g)).sum())
    codes = pd.factorize(g)[0]
    hm = n[H] > 1
    line = {"key": name, "train_groups_gt1": int((g.value_counts() > 1).sum()), "test_match": test_match, "H_match": int(hm.sum()), "H_match_noK0": int((hm & ~inK0).sum())}
    for m in ["gas", "elec", "eui"]:
        y = tr[T[m]].values
        # leave-self-out partner mean of y, y/M, y/A
        res = {}
        for nm, z, scale in [("copy", y, np.ones_like(y)), ("perM", y / M, M), ("perA", y / A, A)]:
            sz = np.bincount(codes, weights=z); ct = np.bincount(codes)
            loo = (sz[codes] - z) / np.maximum(ct[codes] - 1, 1)
            res[nm] = (loo * scale)[H]
        yH = d[f"{m}_y"]; pc = d[f"{m}_corrected"]
        assert np.allclose(yH, y[H])
        sel = hm & ~inK0
        base_sse = ((yH - pc) ** 2).sum()
        tot = len(H)
        for nm, pr in res.items():
            for w in [1.0, 0.5]:
                pn = pc.copy(); pn[sel] = w * pr[sel] + (1 - w) * pc[sel]
                dn = np.sqrt(((yH - pn) ** 2).sum() / tot) / S[m] - np.sqrt(base_sse / tot) / S[m]
                line[f"{m}_{nm}_w{w}"] = round(dn, 6)
        line[f"{m}_rmse_sel_base"] = round(np.sqrt(((yH - pc)[sel] ** 2).mean()) / S[m], 4) if sel.any() else np.nan
        line[f"{m}_rmse_sel_copy"] = round(np.sqrt(((yH - res['copy'])[sel] ** 2).mean()) / S[m], 4) if sel.any() else np.nan
        line[f"{m}_exact_share"] = round(np.mean(np.isclose(yH[sel], res["copy"][sel], rtol=1e-9)), 3) if sel.any() else np.nan
    out.append(line); print(line, flush=True)
pd.DataFrame(out).to_csv("exp/audit/structure/s06_neardup_H.csv", index=False)
# K0 rows on H for reference
print("H rows in K0 groups", int(inK0.sum()))
