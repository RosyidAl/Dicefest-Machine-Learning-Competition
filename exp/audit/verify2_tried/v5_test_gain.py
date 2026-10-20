"""Expected size of the fix-G gain on the TEST rows (verify2_tried), given that the test has fewer extreme gas
predictions than H (30 test rows > 9.5e5 vs 50 expected from the H rate).

Model of the bias above the knot, as in the fix: E[y | g] = g + bt * (g - K) for g > K (bt = 'true' slope).
For a row shifted by d = g - g' (>= 0), the expected squared-error reduction is d * (2 * b - d), b = -bt * (g - K).
1. Check this model reproduces the measured H gain (H rows, bt = H in-sample slope of the corrected residuals).
2. Apply the same per-row formula to the test rows of sub_v19vce (beta of the fix = full-OOF VC teacher -0.6743),
   with bt from (a) H corrected and (b) the full-OOF teacher, and convert to NRMSE with the H base error level.
"""
import numpy as np
import pandas as pd

K, CAP = 9e5, 1e6
SG, SE = 151750.04, 40.65
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
AH = z["A"][z["H"]]
cg, ce, yg, ye = z["gas_corrected"], z["eui_corrected"], z["gas_y"], z["eui_y"]
nH = len(cg)


def slope(p, r):
    h = np.maximum(p - K, 0)
    return float(h @ r / (h @ h))


def expected_reduction(g, a, beta, bt):
    """per-row expected SSE reduction in gas units^2 and in EUI units^2 (implied gas g = e*A for the EUI part)"""
    g2 = np.where(g > K, np.minimum(g + beta * (g - K), CAP), g)
    d = g - g2
    b = np.where(g > K, -bt * (g - K), 0.0)
    return d * (2 * b - d)


bt_H = slope(cg, yg - cg)
pfg = np.load("exp/audit/verify2_tried/full_pf_gas.npy")
tr = pd.read_parquet("cache/train.parquet", columns=["target_macro_gas_kwh"])
bt_F = slope(pfg, tr["target_macro_gas_kwh"].values - pfg)
beta_fix = -0.6743
mse_g, mse_e = np.mean((yg - cg) ** 2), np.mean((ye - ce) ** 2)


def to_total(red_g_sum, red_e_sum, n):
    dg = np.sqrt(mse_g - red_g_sum / n) / SG - np.sqrt(mse_g) / SG
    de = np.sqrt(mse_e - red_e_sum / n) / SE - np.sqrt(mse_e) / SE
    return dg, de, (dg + de) / 5


# 1. model check on H
rg = expected_reduction(cg, AH, beta_fix, bt_H).sum()
re = (expected_reduction(ce * AH, AH, beta_fix, bt_H) / AH ** 2).sum()
dg, de, tot = to_total(rg, re, nH)
g2 = np.where(cg > K, np.minimum(cg + beta_fix * (cg - K), CAP), cg)
ge = ce * AH
e2 = np.where(ge > K, np.minimum(ge + beta_fix * (ge - K), CAP) / AH, ce)
meas = ((np.sqrt(np.mean((yg - g2) ** 2)) - np.sqrt(mse_g)) / SG + (np.sqrt(np.mean((ye - e2) ** 2)) - np.sqrt(mse_e)) / SE) / 5
print(f"bt_H {bt_H:+.4f}  bt_fullOOF {bt_F:+.4f}  beta of the fix {beta_fix:+.4f}")
print(f"H: model-expected TOTAL {tot:+.8f} (gas {dg:+.7f}, eui {de:+.7f}); measured on H with beta {beta_fix}: {meas:+.8f}")
# 2. test rows
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"])
At = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06"])["bld_dim_06"].values
gt, et = sub["target_macro_gas_kwh"].values, sub["target_eui_gas_kwh_per_sqm"].values
nT = len(gt)
sH = np.sum(np.maximum(cg - K, 0) ** 2) / nH
sT = np.sum(np.maximum(gt - K, 0) ** 2) / nT
print(f"mean (g - 9e5)_+^2 per row: H {sH:.4e}, test {sT:.4e}, ratio test/H {sT / sH:.3f}")
for tag, bt in (("bt from H corrected", bt_H), ("bt from full-OOF teacher", bt_F)):
    rg = expected_reduction(gt, At, beta_fix, bt).sum()
    re = (expected_reduction(et * At, At, beta_fix, bt) / At ** 2).sum()
    dg, de, tot = to_total(rg, re, nT)
    print(f"test, {tag}: model-expected TOTAL {tot:+.8f} (gas {dg:+.7f}, eui {de:+.7f})")
# spread of the test gain between public (30%) and private (70%) under a random split of the 105 rows
rng = np.random.default_rng(5)
rg_row = expected_reduction(gt, At, beta_fix, bt_H)
re_row = expected_reduction(et * At, At, beta_fix, bt_H) / At ** 2
pubs, privs = [], []
for _ in range(2000):
    pub = rng.random(nT) < 0.3
    pubs.append(to_total(rg_row[pub].sum(), re_row[pub].sum(), pub.sum())[2])
    privs.append(to_total(rg_row[~pub].sum(), re_row[~pub].sum(), (~pub).sum())[2])
pubs, privs = np.array(pubs), np.array(privs)
print(f"expected-gain spread from which rows land in public vs private (bias model, no noise term): public mean "
      f"{pubs.mean():+.8f} sd {pubs.std():.8f}; private mean {privs.mean():+.8f} sd {privs.std():.8f}")
