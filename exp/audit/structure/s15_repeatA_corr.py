"""(a) For repeated-A rows, is A an exact function of bld_dim_03 (ratio A/b03 constant)? gea_ratio = A/b03 is already a model feature.
(b) Cross-fitted (5-fold on H) linear residual correction with the repeated-A flag and its interactions."""
import numpy as np, pandas as pd
cols = ["bld_dim_06", "bld_dim_03", "bld_dim_01", "bld_dim_08"]
tr = pd.read_parquet("cache/train.parquet", columns=cols); te = pd.read_parquet("cache/test.parquet", columns=cols)
al = pd.concat([tr, te], ignore_index=True); vc = al.bld_dim_06.value_counts(); rep = al.bld_dim_06.map(vc).values > 1
ratio = (al.bld_dim_06 / al.bld_dim_03).values
print("A/b03 quantiles repeated", np.percentile(ratio[rep], [1, 10, 50, 90, 99]).round(5), " others", np.percentile(ratio[~rep], [1, 10, 50, 90, 99]).round(5))
rr = pd.Series(ratio[rep]).round(6).value_counts(); print("most common A/b03 among repeated:", rr.head(5).to_dict())
ro = pd.Series(ratio[~rep]).round(6).value_counts(); print("most common A/b03 among others:", ro.head(5).to_dict())
S = {"gas": 151750.04, "elec": 41725.37, "eui": 40.65}
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]; f = rep[:len(tr)][H].astype(float)
A = d["A"][H]; M = d["M"][H]
for m in ["gas", "eui", "elec"]:
    y = d[f"{m}_y"]; pc = d[f"{m}_corrected"]; r = y - pc
    Z = np.column_stack([f, f * pc, f * np.log(A), f * np.log(M)])
    fo = np.random.default_rng(3).integers(0, 5, len(H)); adj = np.zeros(len(H))
    for k in range(5):
        a, b = fo != k, fo == k
        w = np.linalg.lstsq(Z[a], r[a], rcond=None)[0]; adj[b] = Z[b] @ w
    b0 = np.sqrt(np.mean(r ** 2)) / S[m]; b1 = np.sqrt(np.mean((r - adj) ** 2)) / S[m]
    print(f"{m}: {b0:.5f} -> {b1:.5f} delta {b1 - b0:+.6f} total {(b1 - b0) / 5:+.6f}")
