"""Public-share (30%) variant of G_transfer2.py (a) only. Follow-up of G_transfer.py: uncertainty of the test-level expected gain of the G fix.
(a) stratified bootstrap: per prediction bin (fine bins), resample the realised H per-row SSE changes to the test bin
    counts (full test, and a random 70% 'private' share of the test rows); gas and EUI.
(b) mean h^2 of the full-train OOF (global blend) vs H and test, to put the full-OOF cross-fit gain on the test scale.
Light: chain_H.npz + submission columns + test A + 27 OOF arrays."""
import json

import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65}
CAP, KNOT = 1e6, 9e5
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
AH = A[H]
yH, cH, yHe, cHe = z["gas_y"], z["gas_corrected"], z["eui_y"], z["eui_corrected"]
pP, yP = z["gas_pP"], z["gas_pP"] + z["gas_rP"]
hP = np.maximum(pP - KNOT, 0)
beta = float(hP @ (yP - pP) / (hP @ hP))
G = lambda v: np.minimum(v + beta * np.maximum(v - KNOT, 0), CAP)  # noqa: E731
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["id", GAS, EUI], float_precision="round_trip")
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS])
dup = te[KEY].merge(tr[KEY].drop_duplicates().assign(_d=1), on=KEY, how="left")["_d"].notna().values
At = te["bld_dim_06"].values
sg, geT = sub[GAS].values, sub[EUI].values * At
geH = cHe * AH
nT, nH = len(sg), len(H)
r0g, r0e = np.sqrt(np.mean((yH - cH) ** 2)), np.sqrt(np.mean((yHe - cHe) ** 2))
dsq = (yH - G(cH)) ** 2 - (yH - cH) ** 2
dsqe = np.where(geH > KNOT, (yHe - G(geH) / AH) ** 2 - (yHe - cHe) ** 2, 0.0)
edges = [9e5, 9.25e5, 9.5e5, 9.75e5, 1e6, 1e8]
rng = np.random.default_rng(7)
for share, tag in ((0.3, "public 30%"),):
    res = []
    for _ in range(1000):
        keep = rng.random(nT) < share if share < 1 else np.ones(nT, bool)
        n = keep.sum()
        sg_sum = se_sum = 0.0
        for i in range(len(edges) - 1):
            sH = np.where((cH >= edges[i]) & (cH < edges[i + 1]))[0]
            sHe = np.where((geH >= edges[i]) & (geH < edges[i + 1]))[0]
            kT = int(((sg >= edges[i]) & (sg < edges[i + 1]) & ~dup & keep).sum())
            kTe = int(((geT >= edges[i]) & (geT < edges[i + 1]) & ~dup & keep).sum())
            if kT and len(sH):
                sg_sum += dsq[rng.choice(sH, kT)].sum()
            if kTe and len(sHe):
                se_sum += dsqe[rng.choice(sHe, kTe)].sum()
        res.append(((sg_sum / n) / (2 * r0g) / SIG["gas"] + (se_sum / n) / (2 * r0e) / SIG["eui"]) / 5)
    res = np.array(res)
    print(f"stratified bootstrap ({tag}): TOTAL mean {res.mean():+.8f} sd {res.std():.8f} P(<0) {(res < 0).mean():.3f} "
          f"5..95% [{np.quantile(res, .05):+.8f}, {np.quantile(res, .95):+.8f}]")
