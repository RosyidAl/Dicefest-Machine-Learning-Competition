"""Rare-segment bias scan over every raw feature (+ a few ratios): tails (<=0.1%, 0.1-1%, 99-99.9%, >=99.9% quantiles
of P) and the frequent values of discrete columns. A segment is flagged when the P-side teacher-level residual has
|t| > 4 and the H-side corrected residual has the same sign with |t| > 2. For flagged segments: honest gain of a
multiplicative factor fitted on P, applied to the corrected H predictions (total-score units)."""
import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H = z["P"], z["H"]
skip = {"id", "zone_region_id", "subzone_id", "bld_dim_18", "socio_demog_016"}
import pyarrow.parquet as pq
names = [n for n in pq.read_schema("cache/train.parquet").names if n not in skip and not n.startswith("target_")]
X = pd.read_parquet("cache/train.parquet", columns=names).astype("float32")
X["A_over_M"] = X.bld_dim_06 / X.bld_dim_12
X["E_over_M"] = X.bld_dim_01 / X.bld_dim_12
X["A_over_E"] = X.bld_dim_06 / X.bld_dim_01
X["clean"] = X.bld_dim_02 / X.bld_dim_01
sub = pd.read_parquet("cache/train.parquet", columns=["subzone_id"])["subzone_id"]
X["subzone_rows"] = sub.map(sub.value_counts()).values.astype("float32")
del sub
R = {}
for m in ("gas", "eui", "elec"):
    R[m] = dict(rP=z[f"{m}_rP"], pP=z[f"{m}_pP"], yP=z[f"{m}_rP"] + z[f"{m}_pP"], y=z[f"{m}_y"], c=z[f"{m}_corrected"])
mse = {m: np.mean((R[m]["y"] - R[m]["c"]) ** 2) for m in R}
rows = []
for col in X.columns:
    x = X[col].values
    xP, xH = x[P], x[H]
    bins = {}
    q = np.nanquantile(xP, [0.001, 0.01, 0.99, 0.999])
    bins["<=q0.1%"] = (xP <= q[0], xH <= q[0])
    bins["q0.1-1%"] = ((xP > q[0]) & (xP <= q[1]), (xH > q[0]) & (xH <= q[1]))
    bins["q99-99.9%"] = ((xP >= q[2]) & (xP < q[3]), (xH >= q[2]) & (xH < q[3]))
    bins[">=q99.9%"] = (xP >= q[3], xH >= q[3])
    vals, cnt = np.unique(xP[~np.isnan(xP)], return_counts=True)
    if len(vals) <= 30:
        for v, n in zip(vals, cnt):
            if 50 <= n < 0.2 * len(P):
                bins[f"=={v:g}"] = (xP == v, xH == v)
    for bname, (sp, sh) in bins.items():
        nP, nH = int(sp.sum()), int(sh.sum())
        if nP < 50 or nH < 20 or nP > 0.2 * len(P):
            continue
        for m, d in R.items():
            rp = d["rP"][sp]
            tP = rp.mean() / (rp.std() / np.sqrt(nP) + 1e-12)
            if abs(tP) < 3:
                continue
            rh = d["y"][sh] - d["c"][sh]
            tH = rh.mean() / (rh.std() / np.sqrt(nH) + 1e-12)
            f = (d["yP"][sp] @ d["pP"][sp]) / (d["pP"][sp] @ d["pP"][sp])
            c2 = d["c"].copy()
            c2[sh] *= f
            gain = (np.sqrt(np.mean((d["y"] - c2) ** 2)) - np.sqrt(mse[m])) / SIG[m] / 5
            rows.append(dict(col=col, bin=bname, metric=m, nP=nP, nH=nH, tP=round(float(tP), 1),
                             tH=round(float(tH), 1), factorP=round(float(f), 4), honest_total=float(gain)))
df = pd.DataFrame(rows)
df.to_csv("exp/audit/tried/seg_scan_all_t3.csv", index=False)
flag = df[(np.sign(df.tP) == np.sign(df.tH)) & (df.tH.abs() > 2)].sort_values("honest_total")
pd.set_option("display.width", 200)
print(f"segments with |tP|>3: {len(df)}; confirmed on H (same sign, |tH|>2): {len(flag)}")
print(flag.head(40).to_string(index=False))
print("\nunconfirmed with largest |tP| (H disagrees):")
print(df[~df.index.isin(flag.index)].assign(a=lambda d: -d.tP.abs()).sort_values("a").head(12).drop(columns="a").to_string(index=False))
