"""Floor area A (bld_dim_06, a 17-digit float) repeats exactly across unrelated rows (5274 train groups). What do rows with a
repeated A share, and is a 'repeated-A' flag (computed on train+test features only, no target) informative for the honest residuals on H?"""
import numpy as np, pandas as pd
cols = ["bld_dim_06", "bld_dim_08", "bld_dim_07", "bld_dim_03", "bld_dim_12", "bld_dim_01", "bld_dim_02", "bld_dim_17", "subzone_id"]
tr = pd.read_parquet("cache/train.parquet", columns=cols); te = pd.read_parquet("cache/test.parquet", columns=cols)
al = pd.concat([tr, te], ignore_index=True)
vc = al.bld_dim_06.value_counts(); rep = al.bld_dim_06.map(vc).values > 1
print("rows with repeated A (train+test):", rep.sum(), "share", rep.mean().round(4), "test rows:", rep[len(tr):].sum())
r = al[rep].copy(); g = r.groupby("bld_dim_06")
for c in ["bld_dim_08", "bld_dim_07", "bld_dim_03", "bld_dim_12", "bld_dim_01", "bld_dim_02", "bld_dim_17", "subzone_id"]:
    print(f"  groups with identical {c}: {(g[c].nunique() == 1).mean():.3f}")
print("  A==bld_dim_08 share: repeated", np.mean(r.bld_dim_06 == r.bld_dim_08).round(3), "others", np.mean(al.bld_dim_06[~rep] == al.bld_dim_08[~rep]).round(3))
print("  A/E quantiles repeated", np.percentile(r.bld_dim_06 / r.bld_dim_01, [10, 50, 90]).round(1), "others", np.percentile((al.bld_dim_06 / al.bld_dim_01)[~rep], [10, 50, 90]).round(1))
print("  E quantiles repeated", np.percentile(r.bld_dim_01, [10, 50, 90]), "others", np.percentile(al.bld_dim_01[~rep], [10, 50, 90]))
print("  top repeated A with group E values:", [(round(a, 3), sorted(al.bld_dim_01[al.bld_dim_06 == a].tolist())) for a in vc.index[:4]])
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]
f = rep[:len(tr)][H]
for m in ["gas", "eui", "elec"]:
    y = d[f"{m}_y"]; p = d[f"{m}_corrected"]; lr = np.log(y / p); res = y - p
    print(f"H {m}: flagged {f.sum()}  mean log-res flag {lr[f].mean():+.4f} (se {lr[f].std() / np.sqrt(f.sum()):.4f}) vs {lr[~f].mean():+.4f}; "
          f"mean res flag {res[f].mean():+.3f} vs {res[~f].mean():+.3f}; rmse flag {np.sqrt(np.mean(res[f]**2)):.3f} vs {np.sqrt(np.mean(res[~f]**2)):.3f}")
