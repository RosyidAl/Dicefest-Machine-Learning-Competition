"""Can a twin partner be identified from features? Rank of the true partner among all train rows by feature distance.
Also: are twin rows worse predicted (honest H residuals)?"""
import numpy as np, pandas as pd
T = ["target_macro_gas_kwh", "target_macro_elec_kwh"]
F = ["bld_dim_06", "bld_dim_08", "bld_dim_03", "bld_dim_12", "bld_dim_01", "env_climate_01", "env_climate_03", "env_climate_05", "socio_demog_002"]
tr = pd.read_parquet("cache/train.parquet", columns=F + ["subzone_id"] + T)
gas = tr[T[0]].values
vc = pd.Series(gas).value_counts(); cnt = pd.Series(gas).map(vc).values
X = np.column_stack([np.log(tr[c].clip(lower=1e-3).values) if c.startswith(("bld", "socio")) else tr[c].values for c in F]).astype(np.float64)
X = (X - X.mean(0)) / X.std(0)
tw = np.where(cnt == 2)[0]
order = np.argsort(gas[tw], kind="stable"); tw = tw[order]
pa, pb = tw[0::2], tw[1::2]
rng = np.random.default_rng(0); sel = rng.choice(len(pa), 400, replace=False)
for name, cols in [("size only (A,08,03,M,E,s002)", [0, 1, 2, 3, 4, 8]), ("size+climate", list(range(9))), ("A only", [0])]:
    ranks = []
    for i in sel:
        a, b = pa[i], pb[i]
        d = ((X[:, cols] - X[a, cols]) ** 2).sum(1); d[a] = np.inf
        ranks.append(int((d < d[b]).sum()))
    ranks = np.array(ranks)
    print(name, "partner rank quantiles", np.percentile(ranks, [5, 25, 50, 75]), "top1", np.mean(ranks == 0), "top10", np.mean(ranks < 10))
# oracle-with-gas: if we knew gas to +-r relative, how many train candidates in a similar-size window?
# honest residuals on H for twin rows vs others
d = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H = d["H"]; y = d["gas_y"]; p = d["gas_corrected"]
lr = np.log(y / np.maximum(p, 1)); twH = cnt[H] > 1
print("H twin rows", twH.sum(), "abs log-res twin", np.abs(lr[twH]).mean(), "non-twin", np.abs(lr[~twH]).mean())
print("H gas rmse share from twins", ((y - p)[twH] ** 2).sum() / ((y - p) ** 2).sum(), "row share", twH.mean())
print("mean log-res twin", lr[twH].mean(), "non", lr[~twH].mean())
