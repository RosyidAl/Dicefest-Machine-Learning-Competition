"""Adversarial re-derivation of the S1 small-meter shift on the holdout chain (leakage / optimism lens).

1. Re-derive: LS multiplicative deltas sum(p r)/sum(p^2) per group (M 2-3, M 4) from the P-OOF teacher only, applied
   to the H corrected predictions; change of NRMSE per metric and of the TOTAL score (/5).
2. Leakage probes:
   a. exact duplicate rows (dup_fix key) between P and H, and exact-target twins (same gas / elec value) between the
      small-M H rows and P rows: if an H row's label is replicated in P, the P-fitted delta has partly seen it.
   b. the H gain recomputed after dropping every H row with a P twin and every P row with an H twin.
3. Cluster structure: are the small-M rows concentrated in a few subzones / zones? Cluster-robust (zone, subzone)
   standard errors of the P-side deltas vs the row-level HC0 se; share of small-M H rows whose subzone also holds a
   small-M P row vs the same share for test rows against all train rows.
Only exp/audit/chain_H.npz and a column subset of the cached train/test parquet are read.
"""
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/root/Dicefest"
os.chdir(ROOT)
sys.path.insert(0, "src")
from common import holdout_split  # noqa: E402

OUT = "exp/audit/verify_postproc/adv_leak"
t0 = time.time()
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
YC = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H = z["P"], z["H"]
cols = KEY + ["zone_region_id"] + list(YC.values())
tr = pd.read_parquet("cache/train.parquet", columns=cols)
te = pd.read_parquet("cache/test.parquet", columns=KEY + ["zone_region_id"])
P2, H2 = holdout_split(len(tr))
assert np.array_equal(P2, P) and np.array_equal(H2, H)
M = tr["bld_dim_12"].values.astype(float)
A = tr["bld_dim_06"].values
assert np.array_equal(M, z["M"]) and np.allclose(A, z["A"], rtol=0, atol=0)
for m, c in YC.items():
    assert np.allclose(z[m + "_y"], tr[c].values[H], rtol=1e-12, atol=0)
    assert np.allclose(z[m + "_pP"] + z[m + "_rP"], tr[c].values[P], rtol=1e-9, atol=1e-6)
log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def nrmse(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


GROUPS = {"M2-3": lambda x: (x >= 2) & (x <= 3), "M4": lambda x: x == 4}
MP, MH = M[P], M[H]
say(f"alignment OK; P {len(P)} H {len(H)}; small-M counts P {{M2-3: {GROUPS['M2-3'](MP).sum()}, M4: {(MP == 4).sum()}}}"
    f" H {{M2-3: {GROUPS['M2-3'](MH).sum()}, M4: {(MH == 4).sum()}}}")


def deltas(p, r, Mv, keep=None):
    out = {}
    for g, f in GROUPS.items():
        s = f(Mv) if keep is None else f(Mv) & keep
        out[g] = float(p[s] @ r[s] / (p[s] @ p[s]))
    return out


def apply(c, Mv, d, keep=None):
    q = c.copy()
    for g, f in GROUPS.items():
        s = f(Mv) if keep is None else f(Mv) & keep
        q[s] *= 1 + d[g]
    return q


# ---------------- 1. re-derivation ----------------
say("\n== 1. re-derivation: P-OOF deltas -> H corrected (and H teacher)")
res = {"rederive": {}}
tot_c = tot_t = 0.0
for m in SIG:
    d = deltas(z[m + "_pP"], z[m + "_rP"], MP)
    y, c, t = z[m + "_y"], z[m + "_corrected"], z[m + "_teacher"]
    b_c, b_t = nrmse(y, c, m), nrmse(y, t, m)
    dc = nrmse(y, apply(c, MH, d), m) - b_c
    dt = nrmse(y, apply(t, MH, d), m) - b_t
    tot_c += dc / 5
    tot_t += dt / 5
    res["rederive"][m] = dict(deltas=d, base_corrected=b_c, d_nrmse_corrected=dc, d_nrmse_teacher=dt)
    say(f"   {m:4s} P deltas {{{', '.join(f'{k}: {v:.5f}' for k, v in d.items())}}} | H corrected base {b_c:.6f} "
        f"dNRMSE {dc:+.7f} | H teacher base {b_t:.6f} dNRMSE {dt:+.7f}")
say(f"   TOTAL change on H corrected {tot_c:+.8f} | on H teacher {tot_t:+.8f}   [claim -0.0000458]")
res["rederive"]["total_corrected"], res["rederive"]["total_teacher"] = tot_c, tot_t

# ---------------- 2. leakage probes: duplicates and twins between P and H ----------------
say("\n== 2a. exact duplicate rows (dup_fix key) and exact-target twins between H and P")
keys = pd.MultiIndex.from_frame(tr[KEY].astype(str))
kP = set(keys[P])
dupH = np.array([k in kP for k in keys[H]])
smallH = MH <= 4
say(f"   H rows whose dup key occurs in P: {dupH.sum()} (of which M<=4: {(dupH & smallH).sum()})")
twin = {}
for m in ("gas", "elec"):
    yP = z[m + "_pP"] + z[m + "_rP"]
    yH = z[m + "_y"]
    # exact equality on the raw float values of train (avoid the pP+rP round-off by using the raw column)
    raw = tr[YC[m]].values
    twin[m] = np.isin(raw[H], raw[P])
    s_small = np.isin(raw[H][smallH], raw[P])
    s_small_vs_smallP = np.isin(raw[H][smallH], raw[P][MP <= 4])
    say(f"   {m:4s}: H rows with an exact-value twin in P {twin[m].sum()} ({twin[m].mean():.4f}); "
        f"small-M (M<=4) H rows with a twin anywhere in P {s_small.sum()} of {smallH.sum()}, "
        f"with a twin among small-M P rows {s_small_vs_smallP.sum()}")
anytwinH = twin["gas"] | twin["elec"] | dupH
rawg, rawe = tr[YC["gas"]].values, tr[YC["elec"]].values
kH = set(keys[H])
twinP = np.isin(rawg[P], rawg[H]) | np.isin(rawe[P], rawe[H]) | np.array([k in kH for k in keys[P]])
say(f"   small-M H rows with any twin/dup in P: {(anytwinH & smallH).sum()}; small-M P rows with any twin/dup in H: "
    f"{(twinP & (MP <= 4)).sum()}")
# twins within the full small-M set (any pair of train rows, M<=4) with identical gas
sm = np.where(M <= 4)[0]
vals, cnt = np.unique(rawg[sm], return_counts=True)
say(f"   identical gas values among the {len(sm)} train rows with M<=4: {int((cnt > 1).sum())} values shared by "
    f"{int(cnt[cnt > 1].sum())} rows")

say("\n== 2b. H gain with every twin/duplicate pair between P and H removed (P side and H side)")
tot2 = 0.0
res["no_twins"] = {}
for m in SIG:
    d = deltas(z[m + "_pP"], z[m + "_rP"], MP, keep=~twinP)
    y, c = z[m + "_y"], z[m + "_corrected"]
    q = apply(c, MH, d, keep=~anytwinH)
    dc = nrmse(y, q, m) - nrmse(y, c, m)
    tot2 += dc / 5
    res["no_twins"][m] = dict(deltas=d, d_nrmse=dc)
    say(f"   {m:4s} deltas {{{', '.join(f'{k}: {v:.5f}' for k, v in d.items())}}} dNRMSE {dc:+.7f}")
say(f"   TOTAL without twins/duplicates {tot2:+.8f}")
res["no_twins"]["total"] = tot2

# ---------------- 3. cluster structure ----------------
say("\n== 3. cluster structure of the small-M rows")
sub = tr["subzone_id"].values
zon = tr["zone_region_id"].values
for g, f in GROUPS.items():
    s = f(M)
    vc_sub = pd.Series(sub[s]).value_counts()
    vc_zon = pd.Series(zon[s]).value_counts()
    say(f"   {g}: train rows {s.sum()} in {len(vc_sub)} subzones (max {vc_sub.max()} per subzone) and {len(vc_zon)} "
        f"zones (max {vc_zon.max()}, top-5 zones hold {vc_zon.iloc[:5].sum()} rows)")


def se_cluster(p, r, cl):
    S = p @ p
    d = (p @ r) / S
    u = p * r - d * p * p
    hc0 = np.sqrt(np.sum(u ** 2)) / S
    agg = pd.Series(u).groupby(cl).sum().values
    return d, hc0, np.sqrt(np.sum(agg ** 2)) / S, len(agg)


res["cluster_se"] = {}
for m in SIG:
    pP, rP = z[m + "_pP"], z[m + "_rP"]
    for g, f in GROUPS.items():
        s = f(MP)
        d, hc0, cz, nz = se_cluster(pP[s], rP[s], zon[P][s])
        _, _, cs, ns = se_cluster(pP[s], rP[s], sub[P][s])
        res["cluster_se"][f"{m}_{g}"] = dict(d=d, se_hc0=hc0, se_zone=cz, se_subzone=cs)
        say(f"   P-side {m:4s} {g:4s} d {d:+.4f} se HC0 {hc0:.4f} (t {d / hc0:+.1f}) | zone-clustered {cz:.4f} "
            f"(t {d / cz:+.1f}, {nz} zones) | subzone-clustered {cs:.4f} (t {d / cs:+.1f}, {ns} subzones)")
# subzone sharing: H small rows vs P small rows, and test small rows vs all train small rows
tM = te["bld_dim_12"].values.astype(float)
tsub = te["subzone_id"].values
for g, f in GROUPS.items():
    subP = set(sub[P][f(MP)])
    subT = set(sub[f(M)])
    shH = np.mean([x in subP for x in sub[H][f(MH)]])
    shT = np.mean([x in subT for x in tsub[f(tM)]])
    anyP = set(sub[P])
    shH_any = np.mean([x in anyP for x in sub[H][f(MH)]])
    subAll = set(sub)
    shT_any = np.mean([x in subAll for x in tsub[f(tM)]])
    say(f"   {g}: share of H rows whose subzone holds a same-group P row {shH:.3f} (any P row {shH_any:.3f}) | "
        f"share of test rows whose subzone holds a same-group train row {shT:.3f} (any train row {shT_any:.3f})")
    res[f"subzone_share_{g}"] = dict(H_vs_P_group=shH, test_vs_train_group=shT, H_any=shH_any, test_any=shT_any)

# H-side transfer split by whether the H row's subzone holds a same-group P row
say("\n== 3b. H relative bias (corrected) of small-M rows by subzone sharing with same-group P rows")
for m in SIG:
    y, c = z[m + "_y"], z[m + "_corrected"]
    for g, f in GROUPS.items():
        subP = set(sub[P][f(MP)])
        s = f(MH)
        shared = np.isin(sub[H], np.array(sorted(subP))) & s
        alone = s & ~shared
        out = []
        for nm, ss in (("shared", shared), ("not shared", alone)):
            if ss.sum() > 1:
                b = (c[ss] @ (y[ss] - c[ss])) / (c[ss] @ c[ss])
                out.append(f"{nm} n {ss.sum():3d} LS rel bias {b:+.3f}")
        say(f"   {m:4s} {g:4s} " + " | ".join(out))

ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
say(f"\nmaxrss {ru} kB wall {time.time() - t0:.1f} s")
json.dump(res, open(f"{OUT}/r01_chainH_twins.json", "w"), indent=1, default=float)
open(f"{OUT}/r01_chainH_twins.log", "w").write("\n".join(log) + "\n")
