"""Does the small-meter bias depend on whether the row's subzone is seen by the models? H has 16.8% rows in subzones
without any P row, test only ~8% in subzones without any train row; if the bias lived mostly in unseen rows, the H
estimate would overstate the test gain.
 - H small-M rows split by seen / unseen (subzone has a P row): relative bias, LS factor, gain of S (P factors, s=0.5)
 - P small-M rows split by "subzone has another P row" (proxy for seen in its fold): relative bias at teacher level
 - test small-M rows: share in subzones without any train row
usage: python exp/audit/verify_tried/S_unseen.py
"""
import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M, subn = z["P"], z["H"], z["M"], z["subn"]
MP, MH = M[P], M[H]
GROUPS = (("<=3", lambda v: v <= 3), ("4", lambda v: v == 4))
unseenH = ~np.isin(subn[H], np.unique(subn[P]))
u, cnt = np.unique(subn[P], return_counts=True)
singleP = np.isin(subn[P], u[cnt == 1])
te = pd.read_parquet("cache/test.parquet", columns=["subzone_id", "bld_dim_12"])
subnt = te["subzone_id"].str[4:].astype(int).values
Mt = te["bld_dim_12"].values.astype(float)
unseenT = ~np.isin(subnt, np.unique(subn))
print("test M value counts (M<=5):", {int(v): int((Mt == v).sum()) for v in np.unique(Mt[Mt <= 5])}, flush=True)
for k, g in GROUPS:
    print(f"unseen share {k}: H {unseenH[g(MH)].mean():.3f} (all H {unseenH.mean():.3f}) | test {unseenT[g(Mt)].mean():.3f} "
          f"(all test {unseenT.mean():.3f}) | P singleton-subzone share {singleP[g(MP)].mean():.3f} (all P {singleP.mean():.3f})",
          flush=True)
F0 = {}
for m in SIG:
    pP, yP = z[f"{m}_pP"], z[f"{m}_rP"] + z[f"{m}_pP"]
    F0[m] = {k: float(yP[g(MP)] @ pP[g(MP)] / (pP[g(MP)] @ pP[g(MP)])) for k, g in GROUPS}
    y, c = z[f"{m}_y"], z[f"{m}_corrected"]
    c2 = c.copy()
    for k, g in GROUPS:
        c2[g(MH)] *= 1 + 0.5 * (F0[m][k] - 1)
    di = (y - c2) ** 2 - (y - c) ** 2
    n = len(H)
    mse = np.mean((y - c) ** 2)
    parts = []
    for name, sel in (("seen", ~unseenH), ("unseen", unseenH)):
        for k, g in GROUPS:
            s = g(MH) & sel
            if s.sum() == 0:
                continue
            gain = (np.sqrt(mse + di[s].sum() / n) - np.sqrt(mse)) / SIG[m] / 5
            parts.append(f"{name} {k}: n {int(s.sum())} rel bias {(y[s] - c[s]).sum() / c[s].sum():+.3f} "
                         f"LS f {y[s] @ c[s] / (c[s] @ c[s]):.3f} gain {gain:+.8f}")
    print(f"[H] {m}: " + " | ".join(parts), flush=True)
    rP = yP - pP
    parts = []
    for name, sel in (("multi", ~singleP), ("single", singleP)):
        for k, g in GROUPS:
            s = g(MP) & sel
            if s.sum() == 0:
                continue
            parts.append(f"{name} {k}: n {int(s.sum())} rel bias {rP[s].sum() / pP[s].sum():+.3f} "
                         f"t {rP[s].mean() / (rP[s].std() / np.sqrt(s.sum())):+.1f}")
    print(f"[P] {m}: " + " | ".join(parts), flush=True)
