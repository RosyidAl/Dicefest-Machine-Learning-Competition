"""Groups of rows with an identical full socio_demog vector (incl. the per-row columns): what are they, how many test rows
match a train row on this key, and does a per-EPC / per-meter proportional copy beat the honest prediction on H?"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
names = pq.read_schema("cache/train.parquet").names
socio = sorted(c for c in names if c.startswith("socio_demog"))
socio_x = [c for c in socio if c != "socio_demog_005"]
perrow = ["socio_demog_002", "socio_demog_003", "socio_demog_006", "socio_demog_007", "socio_demog_008", "socio_demog_009", "socio_demog_010",
          "socio_demog_011", "socio_demog_012", "socio_demog_013", "socio_demog_014", "socio_demog_015", "socio_demog_127"]
S = {"gas": 151750.04, "elec": 41725.37, "eui": 40.65}
T = {"gas": "target_macro_gas_kwh", "elec": "target_macro_elec_kwh", "eui": "target_eui_gas_kwh_per_sqm"}
base = ["bld_dim_12", "bld_dim_01", "bld_dim_06", "bld_dim_03", "subzone_id"]
tr = pd.read_parquet("cache/train.parquet", columns=base + socio + list(T.values()))
te = pd.read_parquet("cache/test.parquet", columns=base + socio)
print("per-row socio columns describe (nunique train):", {c: tr[c].nunique() for c in perrow})
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]
def h(df, k): return pd.util.hash_pandas_object(df[k], index=False).values
k0 = h(tr, K0); inK0 = pd.Series(k0).map(pd.Series(k0).value_counts()).values > 1
for kn, key in [("census_x005", socio_x), ("perrow13", perrow), ("perrow13+sub", perrow + ["subzone_id"]), ("perrow13+sub+M+E", perrow + ["subzone_id", "bld_dim_12", "bld_dim_01"])]:
    ktr = h(tr, key); kte = h(te, key)
    vc = pd.Series(ktr).value_counts(); n = pd.Series(ktr).map(vc).values
    multi = n > 1
    tm = pd.Series(kte).isin(vc.index)
    print(f"\n[{kn}] train multi groups {(vc > 1).sum()}, rows in multi {multi.sum()}, of which K0-dup rows {(multi & inK0).sum()}; test rows matching train {int(tm.sum())}")
    codes = pd.factorize(ktr)[0]; ct = np.bincount(codes)
    sel = multi[H] & ~inK0[H]
    print("   H rows matched (not K0):", int(sel.sum()))
    if sel.sum() == 0: continue
    M = tr.bld_dim_12.values; E = tr.bld_dim_01.values; A = tr.bld_dim_06.values
    for m in ["gas", "elec", "eui"]:
        y = tr[T[m]].values; yH = d[f"{m}_y"]; pc = d[f"{m}_corrected"]
        base_rmse = np.sqrt(np.mean((yH - pc) ** 2)) / S[m]
        msg = []
        for nm, den in [("copy", np.ones_like(y)), ("perE", E), ("perM", M), ("perA", A)]:
            z = y / den; sz = np.bincount(codes, weights=z); loo = ((sz[codes] - z) / np.maximum(ct[codes] - 1, 1) * den)[H]
            exact = np.mean(np.isclose(loo[sel], yH[sel], rtol=1e-9))
            for w in [1.0, 0.5]:
                pn = pc.copy(); pn[sel] = w * loo[sel] + (1 - w) * pc[sel]
                msg.append(f"{nm}w{w}:{np.sqrt(np.mean((yH - pn) ** 2)) / S[m] - base_rmse:+.6f}")
            msg.append(f"{nm}_exact:{exact:.3f}")
        sel_rmse_base = np.sqrt(np.mean((yH - pc)[sel] ** 2)) / S[m]
        print(f"   {m}: base rmse on matched {sel_rmse_base:.4f} | " + " ".join(msg))
# inspect a few multi groups of perrow13
ktr = h(tr, perrow); vc = pd.Series(ktr).value_counts(); multi = pd.Series(ktr).map(vc).values > 1
ex = tr.loc[multi, base + perrow[:3] + list(T.values())].assign(k=ktr[multi]).sort_values("k").head(12)
ex.to_csv("exp/audit/structure/s08_examples.csv", index=False)
print(ex.drop(columns="k").round(3).to_string())
