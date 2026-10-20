"""verify2_integrity: own measurement of the 1e6 cap on the holdout chain (exp/audit/chain_H.npz).
The cap has no fitted parameter; the bound is taken either as the round 1e6 or honestly as max(gas) over P rows only.
Reports the change of the TOTAL score (per-target NRMSE change / 5) on all of H, on two different half-splits of H
(sub_n parity, random seed 2027) and its bootstrap sd; teacher (pre-correction) and corrected predictions."""
import json
import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65}
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, subn = z["P"], z["H"], z["A"], z["subn"]
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06"])
g_all = tr[GAS].values
out = {"chain_A_equals_bld_dim_06": bool(np.array_equal(A, tr["bld_dim_06"].values)),
       "gas_y_maxabs_vs_train": float(np.max(np.abs(z["gas_y"] - g_all[H]))),
       "eui_y_maxabs_vs_train": float(np.max(np.abs(z["eui_y"] - tr[EUI].values[H]))),
       "P_H_disjoint_cover": bool(len(np.intersect1d(P, H)) == 0 and len(P) + len(H) == len(tr))}
cP = float(g_all[P].max())  # honest bound: P labels only
out["bound_P_max"] = cP
out["H_gas_max"] = float(g_all[H].max())
AH = A[H]


def nrmse(y, p, s):
    return float(np.sqrt(np.mean((y - p) ** 2)) / s)


def delta_total(y, p, cap, s, idx=None):
    if idx is not None:
        y, p, cap = y[idx], p[idx], cap[idx] if np.ndim(cap) else cap
    q = np.minimum(p, cap)
    return (nrmse(y, q, s) - nrmse(y, p, s)) / 5, int((p > cap).sum())


rng = np.random.default_rng(2027)
halves = {"subn_parity": (subn[H] % 2 == 0), "random2027": rng.random(len(H)) < 0.5}
res = {}
for stage in ["teacher", "corrected"]:
    for c_name, c in [("1e6", 1e6), ("Pmax", cP)]:
        caps = {"gas": np.full(len(H), c), "eui": c / AH}
        for m in ["gas", "eui"]:
            y, p = z[f"{m}_y"], z[f"{m}_{stage}"]
            d, n = delta_total(y, p, caps[m], SIG[m])
            row = {"n_above": n, "base_nrmse": nrmse(y, p, SIG[m]), "delta_total": d}
            for hn, mask in halves.items():
                for side, idx in [("a", np.where(mask)[0]), ("b", np.where(~mask)[0])]:
                    dd, nn = delta_total(y, p, caps[m], SIG[m], idx)
                    row[f"{hn}_{side}"] = {"n_above": nn, "delta_total": dd}
            # bootstrap sd of the total delta (resample H rows)
            sq0, sq1 = (y - p) ** 2, (y - np.minimum(p, caps[m])) ** 2
            bs = []
            br = np.random.default_rng(11)
            for _ in range(2000):
                i = br.integers(0, len(y), len(y))
                bs.append((np.sqrt(sq1[i].mean()) - np.sqrt(sq0[i].mean())) / SIG[m] / 5)
            row["boot_sd"] = float(np.std(bs))
            row["boot_frac_ge0"] = float(np.mean(np.array(bs) >= 0))
            # per-row detail of the capped rows: sign of each row's squared-error change
            k = np.where(p > caps[m])[0]
            row["rows"] = [dict(p=round(float(p[i]), 2), y=round(float(y[i]), 2), cap=round(float(caps[m][i]), 2),
                                sqerr_change=float(sq1[i] - sq0[i])) for i in k[np.argsort(p[k])]]
            row["all_rows_improve"] = bool(np.all(sq1[k] - sq0[k] <= 0))
            res[f"{stage}_{c_name}_{m}"] = row
        res[f"{stage}_{c_name}_total"] = res[f"{stage}_{c_name}_gas"]["delta_total"] + res[f"{stage}_{c_name}_eui"]["delta_total"]
out["results"] = res
# tail calibration on H (corrected): mean of y - p and y/p for high predictions (used for the expected test gain)
for m in ["gas"]:
    y, p = z[f"{m}_y"], z[f"{m}_corrected"]
    for lo in [900_000, 950_000, 1_000_000]:
        k = p > lo
        out[f"H_corrected_{m}_p_gt_{lo}"] = {"n": int(k.sum()), "mean_y_minus_p": float(np.mean(y[k] - p[k])),
                                             "mean_y_over_p": float(np.mean(y[k] / p[k])), "max_y": float(y[k].max())}
json.dump(out, open("exp/audit/verify2_integrity/v2_chainH.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "results"})
for k, v in res.items():
    if isinstance(v, dict):
        print(k, "n", v["n_above"], "base %.7f" % v["base_nrmse"], "dTot %.3e" % v["delta_total"], "sd %.1e" % v["boot_sd"],
              "allimprove", v["all_rows_improve"],
              "halves", {h: "%.2e/%.2e" % (v[f"{h}_a"]["delta_total"], v[f"{h}_b"]["delta_total"]) for h in halves})
    else:
        print(k, "%.3e" % v)
