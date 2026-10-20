"""verify_postproc/adv v04: (a) rebuild the candidate from sub_v19vce with own code (deltas re-derived from the previous
verifier's independent full-train VC OOF, k from all of H, bounds from all train rows) and compare with the claimed file;
(b) check that sub_v19ncapp differs from sub_v19nca only by the same rules (its public LB change is the real-world
test of this claim); (c) public (30%) / private (70%) subset noise of the TOTAL change, by resampling the honest H
per-row changes at test subset sizes, and the likelihood of the observed public change of the analog.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/adv/v04_candidate_lb.py
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/adv"
res = {}

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
nH = len(H)
y = {m: z[f"{m}_y"] for m in MET}
pP = {m: z[f"{m}_pP"] for m in MET}
rP = {m: z[f"{m}_rP"] for m in MET}
VAR = {m: np.linalg.lstsq(np.column_stack([np.ones(len(P)), 1.0 / MP]), (rP[m] / pP[m]) ** 2, rcond=None)[0]
       for m in ("gas", "eui")}
GROUPS = ((2, 3), (4, 4))
gmaxP = (pP["gas"] + rP["gas"]).max()


def tshift(p, s, U):
    b = (U - p) / s
    return -s * np.exp(norm.logpdf(b) - log_ndtr(b))


# ---------------------------------------------------------------- (a) independent rebuild of the candidate
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
Mtr = tr["bld_dim_12"].values.astype(float)
f2 = np.load("exp/audit/verify_postproc/combo/c02_fullvc.npz", allow_pickle=False)
ytr = {m: tr[COL[m]].values for m in MET}
dF = {}
for m in MET:
    p, r = f2[f"{m}_oof"], ytr[m] - f2[f"{m}_oof"]
    dF[m] = [float(p[g] @ r[g] / (p[g] @ p[g])) for g in ((Mtr >= a) & (Mtr <= b) for a, b in GROUPS)]
    res[f"fullOOF_nrmse_{m}"] = float(np.sqrt(np.mean(r ** 2)) / SIGMAS[COL[m]])
fo = json.load(open("exp/audit/postproc/12_fullvc_smallM.json"))
print("full-OOF deltas, own (from verifier c02 OOF)", {m: np.round(v, 5).tolist() for m, v in dF.items()},
      "| auditor json", {m: [round(fo[m]["M<=3"]["ls_delta"], 5), round(fo[m]["M=4"]["ls_delta"], 5)] for m in MET})
kH = {}
for m in ("gas", "eui"):
    p = np.maximum(z[f"{m}_corrected"], 0.0)
    for (a, b) in GROUPS:
        gp, gh = (MP >= a) & (MP <= b), (MH >= a) & (MH <= b)
        p[gh] *= 1 + float(pP[m][gp] @ rP[m][gp] / (pP[m][gp] @ pP[m][gp]))
    x = tshift(p, p * np.sqrt(np.maximum(VAR[m][0] + VAR[m][1] / MH, 1e-8)), gmaxP if m == "gas" else gmaxP / AH)
    kH[m] = float(x @ (y[m] - p) / (x @ x))
print("k (all of H, after S1 with P-OOF deltas)", {m: round(v, 5) for m, v in kH.items()})

ref = tr.groupby(KEY, sort=False)[[GAS, EL, EUI]].first().reset_index()
live = np.isnan(te[KEY].merge(ref, on=KEY, how="left")[GAS].values)
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
Atr = tr["bld_dim_06"].values
gmax, euimin = tr[GAS].max(), tr[EUI].min()
eamin, eamax = (tr[EL] / Atr).min(), (tr[EL] / Atr).max()


def rebuild(sub, deltas):
    new = sub.copy()
    for m in MET:
        p = sub[COL[m]].values.astype(float).copy()
        for j, (a, b) in enumerate(GROUPS):
            g = (Mt >= a) & (Mt <= b) & live
            p[g] *= 1 + deltas[m][j]
        if m in kH:
            x = tshift(p, p * np.sqrt(np.maximum(VAR[m][0] + VAR[m][1] / Mt, 1e-8)), gmax if m == "gas" else gmax / At)
            p = np.where(live, p + kH[m] * x, p)
        lo, hi = {"gas": (euimin * At, np.full(len(At), gmax)), "eui": (np.full(len(At), euimin), gmax / At),
                  "elec": (eamin * At, eamax * At)}[m]
        new[COL[m]] = np.where(live, np.clip(p, lo, hi), p)
    return new


def rd(f):
    return pd.read_csv(f, float_precision="round_trip")


vce, cand = rd("submissions/sub_v19vce.csv"), rd("exp/audit/postproc/sub_v19vce_pp.csv")
sha = hashlib.sha256(open("exp/audit/postproc/sub_v19vce_pp.csv", "rb").read()).hexdigest()
print("candidate sha256", sha, "matches claim:", sha == "8859f0f453e9862df63062abad7c2c6ba985ac6e30bb7c6d81f515a68b47d9c2")
mine = rebuild(vce, dF)
for m in MET:
    c = COL[m]
    print(f"   {m:4s} |own rebuild - candidate|/sigma max {np.max(np.abs(mine[c].values - cand[c].values)) / SIGMAS[c]:.2e}"
          f" | candidate vs vce rows changed {int((cand[c].values != vce[c].values).sum())}")
print("   ids equal", bool((cand["id"].values == vce["id"].values).all()), "| clean/adoption identical",
      bool((cand[[CL, AD]].values == vce[[CL, AD]].values).all()), "| duplicate rows unchanged",
      bool((cand.loc[~live, [GAS, EL, EUI]].values == vce.loc[~live, [GAS, EL, EUI]].values).all()))

# ---------------------------------------------------------------- (b) the public-LB analog: sub_v19nca -> sub_v19ncapp
nca, ncapp = rd("submissions/sub_v19nca.csv"), rd("submissions/sub_v19ncapp.csv")
fon = json.load(open("exp/audit/postproc_nca/12_fullvc_smallM_nca.json"))
dN = {m: [fon[m]["M<=3"]["ls_delta"], fon[m]["M=4"]["ls_delta"]] for m in MET}
mine_n = rebuild(nca, dN)
print("\n(b) analog sub_v19nca -> sub_v19ncapp (public LB 0.24920 -> 0.24913)")
for c in TARGET_COLS:
    diff = ncapp[c].values != nca[c].values
    e = np.max(np.abs(mine_n[c].values - ncapp[c].values)) / SIGMAS[c]
    print(f"   {c:34s} rows changed {int(diff.sum()):6d} | own rule rebuild max err/sigma {e:.2e}")
# how similar are the two edits (nca-pp edit vs vce-pp edit) on the changed rows, in sigma units
for m in MET:
    c = COL[m]
    e1, e2 = (cand[c].values - vce[c].values), (ncapp[c].values - nca[c].values)
    print(f"   {m:4s} edit rms/sigma vce {np.sqrt(np.mean(e1 ** 2)) / SIGMAS[c]:.5f} nca {np.sqrt(np.mean(e2 ** 2)) / SIGMAS[c]:.5f}"
          f" corr {np.corrcoef(e1, e2)[0, 1]:.4f}")

# ---------------------------------------------------------------- (c) subset noise from honest H per-row changes
# honest H predictions of the chain (5-fold, seed 0, S1 P-OOF deltas, k cross-fitted), as in v01
base = {m: np.maximum(z[f"{m}_corrected"], 0.0) for m in MET}
yP = {m: pP[m] + rP[m] for m in MET}
ea = yP["elec"] / AP
LOHI = {"gas": (yP["eui"].min() * AH, np.full(nH, gmaxP)), "eui": (np.full(nH, yP["eui"].min()), gmaxP / AH),
        "elec": (ea.min() * AH, ea.max() * AH)}
after1 = {}
for m in MET:
    q = base[m].copy()
    for (a, b) in GROUPS:
        gp, gh = (MP >= a) & (MP <= b), (MH >= a) & (MH <= b)
        q[gh] *= 1 + float(pP[m][gp] @ rP[m][gp] / (pP[m][gp] @ pP[m][gp]))
    after1[m] = q
honest = {m: np.empty(nH) for m in MET}
for fit, app in KFold(5, shuffle=True, random_state=0).split(np.arange(nH)):
    for m in MET:
        p = after1[m].copy()
        if m in ("gas", "eui"):
            x = tshift(p, p * np.sqrt(np.maximum(VAR[m][0] + VAR[m][1] / MH, 1e-8)), LOHI[m][1])
            p = p + float(x[fit] @ (y[m] - p)[fit] / (x[fit] @ x[fit])) * x
        honest[m][app] = np.clip(p, *LOHI[m])[app]
e0 = {m: ((y[m] - base[m]) / SIGMAS[COL[m]]) ** 2 for m in MET}
e1 = {m: ((y[m] - honest[m]) / SIGMAS[COL[m]]) ** 2 for m in MET}


def tot(idx, center=False):
    d = 0.0
    for m in MET:
        a0 = e0[m][idx].mean()
        dd = e1[m][idx] - e0[m][idx]
        a1 = a0 + (dd - (dmean[m] if center else 0.0)).mean()
        d += np.sqrt(a1) - np.sqrt(a0)
    return d / 5


dmean = {m: float((e1[m] - e0[m]).mean()) for m in MET}
print(f"\n(c) honest H chain TOTAL {tot(np.arange(nH)):+.8f}")
rng = np.random.default_rng(2026)
for name, n in (("public 30%", 55196), ("private 70%", 128791), ("full test", 183987)):
    draws = np.array([tot(rng.integers(0, nH, n)) for _ in range(2000)])
    null = np.array([tot(rng.integers(0, nH, n), center=True) for _ in range(2000)])
    res[f"subset_{name}"] = dict(mean=float(draws.mean()), sd=float(draws.std()), frac_ge0=float(np.mean(draws >= 0)),
                                 null_sd=float(null.std()))
    print(f"   {name:11s} n {n:6d}: change mean {draws.mean():+.8f} sd {draws.std():.8f} P(change >= 0) "
          f"{np.mean(draws >= 0):.3f} | null (effect removed) sd {null.std():.8f}")
    if name == "public 30%":
        lo, hi = -0.00008, -0.00006  # observed analog nca -> ncapp: 0.24920 -> 0.24913 (5-decimal rounding)
        p_alt, p_null = np.mean((draws >= lo) & (draws <= hi)), np.mean((null >= lo) & (null <= hi))
        p_null_le = np.mean(null <= -0.00006)
        print(f"      observed analog change in [{lo}, {hi}]: P | claim-sized effect {p_alt:.3f}, P | no effect "
              f"{p_null:.3f}, P(null <= -0.00006) {p_null_le:.3f}")
        res["lb_analog"] = dict(p_alt=float(p_alt), p_null=float(p_null), p_null_le=float(p_null_le))
json.dump(res, open(f"{OUT}/v04_candidate_lb.json", "w"), indent=1, default=float)
