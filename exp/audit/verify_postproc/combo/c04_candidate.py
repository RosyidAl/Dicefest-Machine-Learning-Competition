"""verify_postproc/combo c04: rebuild the candidate exp/audit/postproc/sub_v19vce_pp.csv independently from
submissions/sub_v19vce.csv and check it is exactly the rule that was evaluated:

  S1 live rows with M in {2,3} / M == 4: p *= 1 + d, d = LS multiplicative bias of the v19vce VC blend on the FULL-train
     OOF (recomputed in c02, independent solver); M == 1 untouched
  S3 gas, eui: p += k x (truncated-normal upper-tail term), k fitted on all of H after S1 with P-OOF deltas,
     (a0, b0) from P-OOF, bound = max train gas (eui: / A)
  S4 clip to full-train bounds; duplicate rows (77) and clean / adoption columns untouched.
Also reports what changed (rows, size, largest changes).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/combo/c04_candidate.py
"""
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/combo"
CAND = "exp/audit/postproc/sub_v19vce_pp.csv"
sha = hashlib.sha256(open(CAND, "rb").read()).hexdigest()
print("candidate sha256", sha, "matches claim:",
      sha == "8859f0f453e9862df63062abad7c2c6ba985ac6e30bb7c6d81f515a68b47d9c2")
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
cand = pd.read_csv(CAND, float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all() and (cand["id"].values == sub["id"].values).all()
ref = tr[KEY].drop_duplicates().assign(_hit=1)
live = te[KEY].merge(ref, on=KEY, how="left")["_hit"].isna().values
print("live rows", int(live.sum()), "duplicates", int((~live).sum()))
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
Y = {"gas": tr[GAS].values, "eui": tr[EUI].values, "elec": tr[EL].values}
f = np.load(f"{OUT}/c02_fullvc.npz")
GROUPS = ((2, 3), (4, 4))
dfull = {}
for m in MET:
    p = np.maximum(f[f"{m}_oof"], 0)
    dfull[m] = []
    for lo, hi in GROUPS:
        g = (M >= lo) & (M <= hi)
        dfull[m].append(float(p[g] @ (Y[m][g] - p[g]) / (p[g] @ p[g])))
print("full-OOF deltas", {m: np.round(v, 4).tolist() for m, v in dfull.items()})
# k and variance model from the holdout chain
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
Pi, Hi = z["P"], z["H"]
MP, MH, AH = M[Pi], M[Hi], A[Hi]
GMAX_P = (z["gas_pP"] + z["gas_rP"]).max()


def tn(p, s, hi):
    b = (hi - p) / s
    return -s * np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))


VM, K = {}, {}
for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    VM[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]
    p = np.maximum(z[f"{m}_corrected"], 0)
    for lo, hi in GROUPS:
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] = p[gH] * (1 + pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    s = p * np.sqrt(np.maximum(VM[m][0] + VM[m][1] / MH, 1e-8))
    x = tn(p, s, GMAX_P if m == "gas" else GMAX_P / AH)
    K[m] = float(x @ (z[f"{m}_y"] - p) / (x @ x))
print("k (all H)", {m: round(v, 4) for m, v in K.items()})
gmax, euimin = Y["gas"].max(), Y["eui"].min()
ea = Y["elec"] / A
mine = sub.copy()
for m in MET:
    p = sub[COL[m]].values.astype(float).copy()
    for (lo, hi), d in zip(GROUPS, dfull[m]):
        g = live & (Mt >= lo) & (Mt <= hi)
        p[g] = p[g] * (1 + d)
    if m in K:
        s = p * np.sqrt(np.maximum(VM[m][0] + VM[m][1] / Mt, 1e-8))
        x = tn(p, s, gmax if m == "gas" else gmax / At)
        p = np.where(live, p + K[m] * x, p)
    lo = {"gas": euimin * At, "eui": np.full(len(At), euimin), "elec": ea.min() * At}[m]
    hi = {"gas": np.full(len(At), gmax), "eui": gmax / At, "elec": ea.max() * At}[m]
    p = np.where(live, np.clip(p, lo, hi), p)
    mine[COL[m]] = p
for c in TARGET_COLS:
    d = np.abs(mine[c].values - cand[c].values) / SIGMAS[c]
    ch = np.abs(cand[c].values - sub[c].values) / SIGMAS[c]
    print(f"{c:34s} |rebuild - candidate|/sigma max {d.max():.2e} | candidate vs sub_v19vce: rows changed "
          f"{int((ch > 0).sum())}, > 0.01 sigma {int((ch > 0.01).sum())}, rms {np.sqrt(np.mean(ch ** 2)):.5f} sigma, "
          f"max {ch.max():.4f} sigma, mean signed {np.mean((cand[c].values - sub[c].values)) / SIGMAS[c]:+.6f} sigma")
print("duplicate rows untouched:", bool(np.array_equal(cand.loc[~live, TARGET_COLS].values, sub.loc[~live, TARGET_COLS].values)))
print("clean / adoption byte-identical:", bool(np.array_equal(cand[[CL, AD]].values, sub[[CL, AD]].values)))
# largest changes
for m in ("eui", "gas", "elec"):
    c = COL[m]
    ch = (cand[c].values - sub[c].values) / SIGMAS[c]
    o = np.argsort(-np.abs(ch))[:5]
    print(f"largest {m} changes: " + "; ".join(f"M {int(Mt[i])} A {At[i]:.0f} {sub[c].values[i]:.4g}->{cand[c].values[i]:.4g} "
                                             f"({ch[i]:+.3f}s)" for i in o))
# changes by M group in the test
for name, g in (("M2-3", (Mt >= 2) & (Mt <= 3)), ("M4", Mt == 4), ("M>=5", Mt >= 5), ("M1", Mt == 1)):
    print(f"test {name}: n {int(g.sum())}, " + ", ".join(
        f"{m} mean rel change {np.mean(cand[COL[m]].values[g] / np.maximum(sub[COL[m]].values[g], 1e-12) - 1):+.4f}"
        for m in MET))
