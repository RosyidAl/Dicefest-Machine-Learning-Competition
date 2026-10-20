"""verify2_integrity: what the 1e6 cap does to the TEST rows of sub_v19vce.
Exact facts: the capped rows and their distance to the bound. Because train gas < 1e6 everywhere (hard truncation),
the true value y <= c, so each capped row's squared error falls by (p - c)(p + c - 2y) >= (p - c)^2: a guaranteed
minimum gain. Expected gain uses E[y | p] calibrations from the holdout chain (corrected predictions) and from the
cross-fitted full OOF (clip_v3). Converted to TOTAL score with a base RMSE taken from the holdout chain."""
import json
import numpy as np
import pandas as pd

GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["id", GAS, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06"])
assert (sub["id"].values == te["id"].values).all()
At = te["bld_dim_06"].values
N = len(sub)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
base = {"gas": 0.2451444, "eui": 0.6485928}  # holdout-chain NRMSE of the corrected prediction (v2_chainH.json)
v3 = np.load("exp/audit/verify2_integrity/clip_v3_fulloof_preds.npz")
tr = pd.read_parquet("cache/train.parquet", columns=[GAS])
g = tr[GAS].values
cal = {}
yH, pH = z["gas_y"], z["gas_corrected"]
k = pH > 1e6
cal["H_ratio_p_gt_1e6"] = float(np.mean(yH[k] / pH[k]))
k = pH > 950_000
cal["H_ratio_p_gt_950k"] = float(np.mean(yH[k] / pH[k]))
po = v3["vc_gas"]
k = po > 1e6
cal["fullOOF_ratio_p_gt_1e6"] = float(np.mean(g[k] / po[k]))
cal["fullOOF_n_p_gt_1e6"] = int(k.sum())
out = {"calibrations_y_over_p": cal}
for m in ["gas", "eui"]:
    p = sub[GAS if m == "gas" else EUI].values
    c = np.full(N, 1e6) if m == "gas" else 1e6 / At
    kk = p > c
    pk, ck = p[kk], c[kk]
    mse0 = (base[m] * SIG[m]) ** 2
    res = {"n_capped": int(kk.sum()), "excess": np.round(pk - ck, 3).tolist()}

    def tot(dsse):
        return float((np.sqrt(mse0 - dsse / N) - np.sqrt(mse0)) / SIG[m] / 5)

    res["guaranteed_min_delta_total(y=c)"] = tot(np.sum((pk - ck) ** 2))
    for name, r in cal.items():
        if not name.startswith(("H_ratio", "fullOOF_ratio")):
            continue
        y = np.minimum(r * pk, ck)  # E[y|p], never above the bound
        res[f"expected_delta_total_{name}"] = tot(np.sum((pk - ck) * (pk + ck - 2 * y)))
    out[m] = res
# public LB view: each capped row is public with prob 0.3 (random 30%); chance that none of the gas rows is public
ng = out["gas"]["n_capped"]
out["prob_no_capped_gas_row_in_public"] = 0.7 ** ng
out["expected_total_change_full_test_range"] = [
    out["gas"]["guaranteed_min_delta_total(y=c)"] + out["eui"]["guaranteed_min_delta_total(y=c)"],
    min(out["gas"][k] for k in out["gas"] if k.startswith("expected")) + min(out["eui"][k] for k in out["eui"] if k.startswith("expected"))]
json.dump(out, open("exp/audit/verify2_integrity/clip_v4_testgain.json", "w"), indent=1)
print(json.dumps(out, indent=1))
