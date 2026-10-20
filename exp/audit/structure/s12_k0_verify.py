"""Verify the exact-duplicate rule K0: (1) on H, copy from another train row is exact for all 5 targets;
(2) in the submission, the 77 matched test rows carry the train targets; (3) test-test K0 duplicates have identical predictions;
(4) do K0 duplicate pairs agree on ALL feature columns?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from common import TARGET_COLS
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet"); te = pd.read_parquet("cache/test.parquet")
feat = sorted(c for c in te.columns if c != "id")
ktr = pd.util.hash_pandas_object(tr[K0], index=False).values; kte = pd.util.hash_pandas_object(te[K0], index=False).values
# (4) feature agreement inside train K0 groups
s = pd.Series(ktr); multi = s.map(s.value_counts()).values > 1
sub = tr.loc[multi, feat].assign(k=ktr[multi])
nun = sub.groupby("k")[feat].nunique(dropna=False)
diffcols = nun.columns[(nun > 1).any()].tolist()
print("train K0 groups", nun.shape[0], "columns that differ inside some group:", diffcols[:10], len(diffcols))
tg = tr.loc[multi, TARGET_COLS].assign(k=ktr[multi]).groupby("k").nunique()
print("targets identical in all K0 groups:", bool((tg == 1).all().all()))
# (1) honest H check
d = np.load("exp/audit/chain_H.npz", allow_pickle=False); H = d["H"]
Hm = multi[H]; print("H rows with K0 partner", int(Hm.sum()))
# (2) submission
sb = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sb.id.values == te.id.values).all()
ref = tr[K0 + TARGET_COLS].drop_duplicates(K0).assign(k=lambda x: pd.util.hash_pandas_object(x[K0], index=False).values).set_index("k")[TARGET_COLS]
hit = np.isin(kte, ref.index.values); print("test rows matching train K0:", int(hit.sum()))
diff = np.abs(sb.loc[hit, TARGET_COLS].values - ref.loc[kte[hit]].values).max(axis=0)
print("max |sub - train target| on matched rows per target:", diff)
# (3) test-test duplicates without train match
st = pd.Series(kte); tm = (st.map(st.value_counts()).values > 1) & ~hit
print("test rows in test-only K0 groups:", int(tm.sum()))
if tm.any():
    g = sb.loc[tm, TARGET_COLS].assign(k=kte[tm]).groupby("k")
    spread = (g.max() - g.min()).max()
    print("max within-group prediction spread per target:", spread.round(6).to_dict())
    fe = te.loc[tm, feat].assign(k=kte[tm]).groupby("k")[feat].nunique(dropna=False)
    print("feature columns differing in test-only groups:", fe.columns[(fe > 1).any()].tolist()[:10])
