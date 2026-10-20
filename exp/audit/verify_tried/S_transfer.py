"""Transfer checks for the small-meter fix S (bld_dim_12 <= 4) from the holdout chain to the test set.

 a. test rows per group, overlap with the 77 duplicate rows (dup_fix key)
 b. does the deployed residual correction already move small-M test rows (test: sub_v19vce vs the VC teacher
    exp/pl/teacher_deploy.npz) more than it moves small-M H rows (corrected vs teacher)? If so, part of the bias would
    already be absorbed on test and the P factor would over-correct.
 c. EUI: the gain lives in sum(c^2) of the group rows; compare the prediction scale of small-M rows on H and test
 d. model-based expected change (bias b assumed, factor g applied): delta SSE = sum c^2 [(g - b)^2 - (1 - b)^2]
    on H rows and on test rows (RMSE base taken from H), for several b (P, H-corrected, full-OOF global)
 e. the audit candidate exp/audit/tried/sub_v19vce_GS_audit.csv: rows changed by S and the multipliers used
 f. the H rows that carry the EUI gain (top 10 by SSE reduction)
usage: python exp/audit/verify_tried/S_transfer.py
"""
import json

import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
COL = {"gas": "target_macro_gas_kwh", "elec": "target_macro_elec_kwh", "eui": "target_eui_gas_kwh_per_sqm"}
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
MH, AH = M[H], A[H]
prm = json.load(open("exp/audit/tried/combo_params_P.json"))["factors"]
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
tr = pd.read_parquet("cache/train.parquet", columns=KEY)
dup = te[KEY].merge(tr.drop_duplicates(KEY).assign(_d=1), on=KEY, how="left")["_d"].notna().values
del tr
Mt, At = te["bld_dim_12"].values.astype(float), te["bld_dim_06"].values
GT = {"<=3": Mt <= 3, "4": Mt == 4}
GH = {"<=3": MH <= 3, "4": MH == 4}
print("[a] test rows:", {k: int(v.sum()) for k, v in GT.items()}, "dup rows:", int(dup.sum()),
      "dup in groups:", int((dup & (Mt <= 4)).sum()), flush=True)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
tdep = np.load("exp/pl/teacher_deploy.npz", allow_pickle=False)
for m in SIG:
    t, s = tdep[m], sub[COL[m]].values
    yH, cH, tH = z[f"{m}_y"], z[f"{m}_corrected"], z[f"{m}_teacher"]
    msg = []
    for k in GT:
        st, sh = GT[k] & ~dup, GH[k]
        msg.append(f"{k}: test corr/teacher {(s[st] - t[st]).sum() / t[st].sum():+.4f} (rms rel "
                   f"{np.sqrt(np.mean(((s[st] - t[st]) / t[st]) ** 2)):.4f}) vs H {(cH[sh] - tH[sh]).sum() / tH[sh].sum():+.4f} "
                   f"(rms rel {np.sqrt(np.mean(((cH[sh] - tH[sh]) / tH[sh]) ** 2)):.4f})")
    allrel_t = np.sqrt(np.mean(((s[~dup] - t[~dup]) / np.maximum(t[~dup], 1e-9)) ** 2))
    allrel_h = np.sqrt(np.mean(((cH - tH) / np.maximum(tH, 1e-9)) ** 2))
    print(f"[b] {m}: " + " || ".join(msg) + f" || all rows rms rel corr test {allrel_t:.4f} H {allrel_h:.4f}", flush=True)

# c + d: prediction scale and model-based expected change
OOF_GLOBAL = {"gas": {"<=3": 1.2264, "4": 1.0731}, "eui": {"<=3": 1.2351, "4": 1.0648},
              "elec": {"<=3": 1.2834, "4": 1.1527}}  # exp/audit/tried/params_full_oof.log
for m in SIG:
    yH, cH = z[f"{m}_y"], z[f"{m}_corrected"]
    s = sub[COL[m]].values
    mse_h = np.mean((yH - cH) ** 2)
    hc = {k: cH[GH[k]] for k in GH}
    tc = {k: s[GT[k] & ~dup] for k in GT}
    print(f"[c] {m}: H group pred mean/q90/max " + ", ".join(f"{k} {v.mean():.4g}/{np.quantile(v, .9):.4g}/{v.max():.4g}"
                                                         for k, v in hc.items())
          + " | test " + ", ".join(f"{k} {v.mean():.4g}/{np.quantile(v, .9):.4g}/{v.max():.4g}" for k, v in tc.items())
          + f" | sum c^2 per row (H, test): " + ", ".join(f"{k} {np.sum(hc[k] ** 2) / len(H):.4g}, {np.sum(tc[k] ** 2) / len(s):.4g}"
                                                          for k in GH), flush=True)
    bsrc = {"P": {k: prm[m][k] for k in GH},
            "Hcorr": {k: float(yH[GH[k]] @ cH[GH[k]] / (cH[GH[k]] @ cH[GH[k]])) for k in GH},
            "OOFglobal": OOF_GLOBAL[m]}
    for name, b in bsrc.items():
        for sh in (0.5, 1.0):
            out = []
            for setname, cs, n in (("H", hc, len(H)), ("test", tc, len(s))):
                dsse = sum(np.sum(cs[k] ** 2) * ((1 + sh * (prm[m][k] - 1) - b[k]) ** 2 - (1 - b[k]) ** 2) for k in GH)
                d = (np.sqrt(mse_h + dsse / n) - np.sqrt(mse_h)) / SIG[m] / 5
                out.append(f"{setname} {d:+.8f}")
            print(f"[d] {m} true bias = {name} {({k: round(v, 3) for k, v in b.items()})}, g = P factor, s={sh}: "
                  f"expected total {', '.join(out)}", flush=True)

# e: the audit candidate
cand = pd.read_csv("exp/audit/tried/sub_v19vce_GS_audit.csv", float_precision="round_trip")
assert (cand["id"].values == sub["id"].values).all()
for m in SIG:
    a, b = sub[COL[m]].values, cand[COL[m]].values
    for k in GT:
        st = GT[k] & ~dup
        r = b[st] / a[st]
        print(f"[e] {m} {k}: candidate/base ratio min {r.min():.5f} max {r.max():.5f} (intended {1 + 0.5 * (prm[m][k] - 1):.5f})",
              flush=True)
    other = ~(Mt <= 4)
    ch = (a != b) & other
    print(f"[e] {m}: rows changed outside M<=4: {int(ch.sum())} (G fix), dup rows changed: {int(((a != b) & dup).sum())}",
          flush=True)
for c in ["target_clean_building_ratio", "target_electric_heating_adoption"]:
    assert (cand[c].values == sub[c].values).all()

# f: H rows carrying the EUI gain
m = "eui"
yH, cH = z[f"{m}_y"], z[f"{m}_corrected"]
c2 = cH.copy()
for k in GH:
    c2[GH[k]] *= 1 + 0.5 * (prm[m][k] - 1)
di = (yH - c2) ** 2 - (yH - cH) ** 2
idx = np.argsort(di)[:12]
print("[f] top EUI rows: M, A, y, corrected, fixed, dSSE, resid/corrected")
for i in idx:
    print(f"    M={MH[i]:.0f} A={AH[i]:9.1f} y={yH[i]:8.2f} c={cH[i]:8.2f} c'={c2[i]:8.2f} dSSE={di[i]:+9.1f} "
          f"rel={(yH[i] - cH[i]) / cH[i]:+.3f}")
g = MH <= 4
print(f"[f] EUI group rows: mean y {yH[g].mean():.2f} mean c {cH[g].mean():.2f}; median rel resid "
      f"{np.median((yH[g] - cH[g]) / cH[g]):+.4f}; share of rows with y > c: {(yH[g] > cH[g]).mean():.3f}; "
      f"all H rows share y > c {(yH > cH).mean():.3f}")
for k in GH:
    s_ = GH[k]
    lr = np.log(np.maximum(yH[s_], 1e-6) / cH[s_])
    print(f"[f] EUI {k}: median y/c {np.exp(np.median(lr)):.4f}, mean log(y/c) {lr.mean():+.4f} (t {lr.mean() / (lr.std() / np.sqrt(s_.sum())):+.2f}),"
          f" sum(y)/sum(c) {yH[s_].sum() / cH[s_].sum():.4f}")
