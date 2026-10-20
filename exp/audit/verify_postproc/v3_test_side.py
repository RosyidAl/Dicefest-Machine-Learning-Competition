"""verify_postproc v3: does the S3 rule see the same situation on test as on H?

(a) density of predictions near / above the gas cap (per 10k rows, by standardized distance) on test (sub_v19vce, final
    corrected predictions), on H (corrected chain) and on the full-train OOF teacher. If test were NOT truncated at
    1e6 (e.g. cleaning applied to train only), test would show a clear excess of near-cap / above-cap predictions.
(b) the S3 change on test vs H (rows per 10k moved by > 0.01 / 0.1 sigma, rms change), with the claimed constants.
(c) the expected count of rows above 1e6 if the gas tail just continued past the cap (geometric extrapolation of the
    train histogram) -> how big an excess the test check in (a) should be able to see.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/v3_test_side.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
COL = {"gas": GAS, "eui": EUI}
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
ref = tr.groupby(KEY, sort=False)[TARGET_COLS].first().reset_index()
isdup = ~np.isnan(te[KEY].merge(ref, on=KEY, how="left")[TARGET_COLS].values[:, 0])
print("duplicates", int(isdup.sum()))
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
print("test M min", Mt.min(), "rows M<2:", int((Mt < 2).sum()), " train M min", tr["bld_dim_12"].min())
T_all = float(tr[GAS].max())
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
f = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
MP, MH = Mall[P], Mall[H]
T_P = float((z["gas_pP"] + z["gas_rP"]).max())
SD = {}
for m in SIG:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


def feats(m, p, M, Ar, T):
    p = np.clip(p, 0, None)
    s = np.maximum(p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / M, 1e-8, None)), 1e-12)
    hi = T if m == "gas" else T / Ar
    return (hi - p) / s, tn_hi(p, s, hi)


kH = {}
for m in SIG:
    yH, pH = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    _, x = feats(m, pH, MH, A[H], T_P)
    kH[m] = float(x @ (yH - pH) / (x @ x))
print("k (all H corrected):", {m: round(v, 4) for m, v in kH.items()}, "T_P", round(T_P, 1), "T_all", round(T_all, 1))

EDGES = ((-99, -1), (-1, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3))
print("== (a) predictions per 10k rows by standardized distance to the cap")
for m in SIG:
    sets = {
        "test final": feats(m, sub[COL[m]].values[~isdup], Mt[~isdup], At[~isdup], T_all)[0],
        "H corrected": feats(m, z[f"{m}_corrected"], MH, A[H], T_P)[0],
        "H teacher": feats(m, z[f"{m}_teacher"], MH, A[H], T_P)[0],
        "full OOF teacher": feats(m, f[f"{m}_p"], Mall, A, T_all)[0],
    }
    for name, dist in sets.items():
        cnt = [1e4 * np.mean((dist >= lo) & (dist < hi)) for lo, hi in EDGES]
        print(f"   {m:4s} {name:17s} " + " ".join(f"[{lo},{hi}) {c:5.2f}" for (lo, hi), c in zip(EDGES, cnt)))
print("   raw gas level shares per 10k rows (p > 0.8M / 0.9M / 0.95M / 1.0M):")
for name, p in (("test final", sub[GAS].values[~isdup]), ("H corrected", z["gas_corrected"]),
                ("full OOF teacher", f["gas_p"]), ("train truth", tr[GAS].values)):
    print(f"   {name:17s} " + " ".join(f"{1e4 * np.mean(p > t):6.2f}" for t in (8e5, 9e5, 9.5e5, 1e6)))

print("== (b) S3 change, test vs H (claimed constants)")
for m in SIG:
    _, xt = feats(m, sub[COL[m]].values, Mt, At, T_all)
    ch_t = np.where(isdup, 0.0, kH[m] * xt) / SIG[m]
    _, xh = feats(m, z[f"{m}_corrected"], MH, A[H], T_P)
    ch_h = kH[m] * xh / SIG[m]
    for name, ch in (("test", ch_t), ("H", ch_h)):
        print(f"   {m:4s} {name:4s}: per 10k |d|>0.01s {1e4 * np.mean(np.abs(ch) > 0.01):5.1f}  |d|>0.1s"
              f" {1e4 * np.mean(np.abs(ch) > 0.1):5.2f}  rms {np.sqrt(np.mean(ch ** 2)):.5f}s  min {ch.min():+.3f}s"
              f"  max {ch.max():+.2e}s")
    j = np.argmin(ch_t)
    print(f"   {m} largest test change: p {sub[COL[m]].values[j]:.1f} -> {sub[COL[m]].values[j] + kH[m] * xt[j]:.1f} (M {Mt[j]:.0f}, A {At[j]:.0f})")

print("== (c) how many rows would lie above 1e6 if the tail continued (geometric extrapolation of 10k-kWh bins)")
g = tr[GAS].values
h = np.histogram(g, bins=np.arange(8e5, 1e6 + 1, 1e4))[0]
ratio = (h[-5:].mean() / h[:5].mean()) ** (1 / 15)
above = h[-5:].mean() * ratio / (1 - ratio)
print(f"   last-5-bin mean {h[-5:].mean():.1f} per 10k kWh, per-bin decay {ratio:.3f} -> ~{above:.0f} train rows "
      f"({1e4 * above / len(g):.1f} per 10k rows) would be above 1e6; test analogue ~{above * len(te) / len(g):.0f} rows")
