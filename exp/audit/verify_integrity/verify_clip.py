"""Adversarial re-derivation of the 'hard clip at the 1e6 gas bound' finding (integrity lens).

Checks:
 1. H effect re-derived from chain_H.npz with a bound taken from P rows only (no H label used for the bound).
 2. Per-row sign: every clipped H row must lose error (truth < bound); bootstrap / split-half spread of the delta.
 3. Does the truncation transfer to test? Compare the high tail of (a) an OOF NNLS gas blend on train vs its test
    prediction, (b) floor area A and meters M, train vs test. Same truncation => test/train count ratio ~ 0.4286.
 4. Expected test-side gain for the 7 gas / 8 EUI submitted rows above the bound, using E[y | p] from H.
 5. Safety of the proposed code: NaN / zero / negative floor area in test (1e6 / A).
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/verify_clip.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

SIG = dict(gas=151750.04, eui=40.65, elec=41725.37)
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
OUT = "exp/audit/verify_integrity/verify_clip.json"
rng = np.random.default_rng(0)
out = {}


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
tr = pd.read_parquet("data/cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12"])
assert np.array_equal(tr["bld_dim_06"].values, A)
ygas = tr[GAS].values
out["train_n"] = len(tr)
out["max_gas_P"] = float(ygas[P].max())
out["max_gas_H"] = float(ygas[H].max())
out["n_gas_ge_1e6_P"] = int((ygas[P] >= 1e6).sum())
out["n_gas_ge_1e6_H"] = int((ygas[H] >= 1e6).sum())

# 1-2. H effect, bound from P only (round 1e6 and the P max itself)
AH = A[H]
for cap_name, cap in (("1e6", 1e6), ("Pmax", float(ygas[P].max()))):
    res = {}
    for m in ("gas", "eui"):
        y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
        hi = cap if m == "gas" else cap / AH
        pc = np.minimum(p, hi)
        msk = p > hi
        row_gain = (p - y) ** 2 - (pc - y) ** 2  # >= 0 means clip helps that row
        d = (nr(y, pc, m) - nr(y, p, m)) / 5
        # bootstrap over H rows (resample rows, recompute delta)
        bs = []
        for _ in range(400):
            i = rng.integers(0, len(y), len(y))
            bs.append((nr(y[i], pc[i], m) - nr(y[i], p[i], m)) / 5)
        # split-half: two random halves of H
        perm = rng.permutation(len(y))
        h1, h2 = perm[: len(y) // 2], perm[len(y) // 2:]
        res[m] = dict(n_above=int(msk.sum()), base=nr(y, p, m), clipped=nr(y, pc, m), delta_total=d,
                      rows_hurt=int((row_gain < 0).sum()), min_row_gain=float(row_gain[msk].min()) if msk.any() else 0.0,
                      truth_max_over_bound=float((y / hi).max()),
                      boot_mean=float(np.mean(bs)), boot_sd=float(np.std(bs)), boot_p_ge0=float(np.mean(np.array(bs) >= 0)),
                      half1=(nr(y[h1], pc[h1], m) - nr(y[h1], p[h1], m)) / 5,
                      half2=(nr(y[h2], pc[h2], m) - nr(y[h2], p[h2], m)) / 5,
                      excess_above=sorted((p - hi)[msk].round(3).tolist()))
    out[f"H_cap_{cap_name}"] = res
    print(cap_name, json.dumps({m: {k: v for k, v in r.items() if k != "excess_above"} for m, r in res.items()}))

# 3a. train-gas tail (bins of 50k) and a geometric extrapolation of what an untruncated tail would hold above 1e6
edges = np.arange(6.0e5, 1.0e6 + 1, 5.0e4)
cnt = np.histogram(ygas, bins=edges)[0]
out["train_gas_bins_50k_from_600k"] = cnt.tolist()
r = float(np.exp(np.polyfit(np.arange(len(cnt)), np.log(cnt), 1)[0]))
out["tail_decay_per_50k"] = r
out["untruncated_expected_train_rows_ge_1e6"] = float(cnt[-1] * r / (1 - r))
out["untruncated_expected_test_rows_ge_1e6"] = float(cnt[-1] * r / (1 - r) * 183987 / len(tr))

# 3b. OOF NNLS gas blend (global weights, EUI scale x A as in the pipeline) -> tail counts train OOF vs test pred
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
te = pd.read_parquet("data/cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
Po = np.column_stack([np.load(f"exp/oof/{k}.npy") for k in mem])
Pt = np.column_stack([np.load(f"exp/pred/{k}.npy") for k in mem])
w, _ = nnls(Po * A[:, None], tr[EUI].values * A)
go, gt = (Po @ w) * A, (Pt @ w) * At
del Po, Pt
out["oof_blend_gas_nrmse_insample_w"] = nr(ygas, go, "gas")
ratio_expected = 183987 / len(tr)
out["test_train_ratio_expected"] = ratio_expected
tails = {}
for t in (7e5, 8e5, 9e5, 9.5e5, 1e6, 1.05e6):
    ntr, nte = int((go > t).sum()), int((gt > t).sum())
    tails[f"{t:.0f}"] = dict(train_oof=ntr, test=nte, ratio=nte / max(ntr, 1), test_expected_if_same=ntr * ratio_expected)
out["oof_blend_tail_counts"] = tails
# true-gas distribution of train rows whose OOF blend prediction is near / above the bound (for E[y | p])
for lo, hi_ in ((9.5e5, 1e6), (1e6, 1.1e6)):
    msk = (go > lo) & (go <= hi_)
    out[f"oof_rows_pred_{lo:.0f}_{hi_:.0f}"] = dict(n=int(msk.sum()), mean_pred=float(go[msk].mean()),
                                                     mean_y=float(ygas[msk].mean()), max_y=float(ygas[msk].max()))
# 3c. size tails (feature only, no target): A and M
qs = (0.999, 0.9999)
size = {}
for name, a_tr, a_te in (("A", A, At), ("M", tr["bld_dim_12"].values.astype(float), Mt)):
    for q in qs:
        thr = float(np.quantile(a_tr, q))
        ntr, nte = int((a_tr > thr).sum()), int((a_te > thr).sum())
        size[f"{name}_q{q}"] = dict(thr=thr, train=ntr, test=nte, ratio=nte / max(ntr, 1))
out["size_tail_counts"] = size

# 4. submitted file: rows above the bound and their expected gain with E[y | p] from the H chain
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
g, e = sub[GAS].values, sub[EUI].values
out["test_A_nan"] = int(np.isnan(At).sum())
out["test_A_le0"] = int((At <= 0).sum())
out["test_A_min"] = float(np.nanmin(At))
yH, pH = z["gas_y"], np.clip(z["gas_corrected"], 0, None)
yE, pE = z["eui_y"], np.clip(z["eui_corrected"], 0, None)
near_g = pH > 9.5e5
gap_g = float(np.mean(1e6 - yH[near_g]))  # E[c - y | p > 950k] on H
near_e = pE > 0.95e6 / AH
gap_e = float(np.mean(1e6 / AH[near_e] - yE[near_e]))
mg = g > 1e6
me = e > 1e6 / At
n = len(sub)
mse_g = (0.24514 * SIG["gas"]) ** 2
mse_e = (0.64859 * SIG["eui"]) ** 2
dmse_g_lb = np.sum((g[mg] - 1e6) ** 2) / n
dmse_g_exp = np.sum((g[mg] - 1e6) * (g[mg] - 1e6 + 2 * gap_g)) / n
ce = 1e6 / At[me]
dmse_e_lb = np.sum((e[me] - ce) ** 2) / n
dmse_e_exp = np.sum((e[me] - ce) * (e[me] - ce + 2 * gap_e)) / n


def dtot(mse, dmse, m):
    return float((np.sqrt(mse - dmse) - np.sqrt(mse)) / SIG[m] / 5)


out["test_side"] = dict(n_gas_above=int(mg.sum()), n_eui_above=int(me.sum()), H_gap_gas=gap_g, H_gap_n=int(near_g.sum()),
                        H_gap_eui=gap_e, H_gap_eui_n=int(near_e.sum()),
                        gas_delta_total_lower_bound=dtot(mse_g, dmse_g_lb, "gas"),
                        gas_delta_total_expected=dtot(mse_g, dmse_g_exp, "gas"),
                        eui_delta_total_lower_bound=dtot(mse_e, dmse_e_lb, "eui"),
                        eui_delta_total_expected=dtot(mse_e, dmse_e_exp, "eui"),
                        rows_gas_both_cols=int((mg & me).sum()))
out["test_side"]["total_expected"] = out["test_side"]["gas_delta_total_expected"] + out["test_side"]["eui_delta_total_expected"]
print(json.dumps({k: v for k, v in out.items() if not k.startswith("H_cap")}, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
