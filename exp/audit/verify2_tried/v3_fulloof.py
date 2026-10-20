"""Independent full-data check (verify2_tried) of the gas upper-bound fix G at teacher level.

Rebuilds the production VC teacher (final/src/postprocess.blend_preds, spec from exp/v19vce/resid_knn.json, gas members
from exp/v19vce/members.yaml) on the FULL train OOF of the v19 members, i.e. exactly the `pf` the pipeline action would
fit beta on, and the matching test-side teacher `pa` (checked against exp/pl/teacher_deploy.npz).
Then:
  * bias of the full-OOF teacher by prediction bin;
  * beta as the action would compute it (all train rows);
  * honest cross-fits on the full OOF (beta fitted on other rows): 5-fold with seeds 7..26 and 50 random halvings.
Saves pf/pa to exp/audit/verify2_tried/ (small .npy files) for v4.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "final/src")
from postprocess import blend_preds  # noqa: E402

OUT = "exp/audit/verify2_tried"
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
SIG = {"gas": 151750.04, "eui": 40.65}
K, CAP = 9e5, 1e6
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12"])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
yg, ye = tr[GAS].values, tr[EUI].values
del tr, te
members = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
spec = json.load(open("exp/v19vce/resid_knn.json"))["vc"]
print(f"{len(members)} gas-family members; vc spec {spec['gas']} / {spec['eui']}", flush=True)
files = {m: f"{OUT}/full_{m}.npy" for m in ("pf_gas", "pa_gas", "pf_eui", "pa_eui")}
if not all(os.path.exists(f) for f in files.values()):
    Pf = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in members])
    Pa = np.column_stack([np.load(f"exp/pred/{r}.npy") for r in members])
    for metric in ("gas", "eui"):
        pf, pa = blend_preds(Pf, Pa, ye, metric, A, At, spec[metric], M, Mt)
        np.save(files[f"pf_{metric}"], pf)
        np.save(files[f"pa_{metric}"], pa)
        print(f"blend {metric} done", flush=True)
    del Pf, Pa
pf_g, pa_g, pf_e, pa_e = (np.load(files[k]) for k in ("pf_gas", "pa_gas", "pf_eui", "pa_eui"))
td = np.load("exp/pl/teacher_deploy.npz", allow_pickle=False)
for k, v in (("gas", pa_g), ("eui", pa_e)):
    ref = td[k]
    print(f"test teacher vs teacher_deploy.npz[{k}]: max abs diff {np.max(np.abs(np.clip(v, 0, None) - ref)):.3e} "
          f"(ref max {ref.max():.1f})")


def nrmse(y, p, s):
    return np.sqrt(np.mean((y - p) ** 2)) / s


print(f"full-OOF VC teacher NRMSE gas {nrmse(yg, pf_g, SIG['gas']):.6f} eui {nrmse(ye, pf_e, SIG['eui']):.6f}")
print(f"train rows pf_gas > 9e5: {np.sum(pf_g > K)}, > 1e6: {np.sum(pf_g > CAP)}, max {pf_g.max():.0f}; "
      f"implied eui*A > 9e5: {np.sum(pf_e * A > K)}")
print(f"test rows teacher gas > 9e5: {np.sum(pa_g > K)}, > 1e6: {np.sum(pa_g > CAP)}, max {pa_g.max():.0f}; "
      f"implied eui*A > 9e5: {np.sum(pa_e * At > K)}")
print("full-OOF teacher mean residual by gas-prediction bin:")
for lo, hi in [(7e5, 8e5), (8e5, 8.5e5), (8.5e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 3e6)]:
    m = (pf_g >= lo) & (pf_g < hi)
    r = yg[m] - pf_g[m]
    print(f"  [{lo:9.0f},{hi:9.0f}) n={m.sum():5d} resid {r.mean():+9.0f} (se {r.std(ddof=1) / np.sqrt(m.sum()):6.0f})")


def beta_fit(p, r):
    h = np.maximum(p - K, 0.0)
    return float(h @ r / (h @ h))


b_all = beta_fit(pf_g, yg - pf_g)
print(f"beta as the pipeline action would fit it (all train rows, VC teacher pf): {b_all:+.4f}")
b_e = beta_fit(pf_e * A, (ye - pf_e) * A)
print(f"(reference) beta fitted on the EUI teacher's implied gas: {b_e:+.4f}")


def apply(g, e, a, beta):
    g2 = np.where(g > K, np.minimum(g + beta * (g - K), CAP), g)
    ge = e * a
    e2 = np.where(ge > K, np.minimum(ge + beta * (ge - K), CAP) / a, e)
    return g2, e2


base_g, base_e = nrmse(yg, pf_g, SIG["gas"]), nrmse(ye, pf_e, SIG["eui"])


def crossfit(fold, nf):
    g2, e2 = pf_g.copy(), pf_e.copy()
    bs = []
    for f in range(nf):
        a, b = fold != f, fold == f
        beta = beta_fit(pf_g[a], yg[a] - pf_g[a])
        bs.append(beta)
        g2[b], e2[b] = apply(pf_g[b], pf_e[b], A[b], beta)
    dg, de = nrmse(yg, g2, SIG["gas"]) - base_g, nrmse(ye, e2, SIG["eui"]) - base_e
    return dg, de, (dg + de) / 5, bs


n = len(yg)
for nf, seeds, tag in [(5, range(7, 27), "5-fold"), (2, range(500, 550), "split-half")]:
    res = [crossfit(np.random.default_rng(s).permutation(n) % nf, nf) for s in seeds]
    tot = np.array([r[2] for r in res])
    dg = np.array([r[0] for r in res])
    de = np.array([r[1] for r in res])
    bet = np.concatenate([r[3] for r in res])
    print(f"full-OOF {tag} cross-fit ({len(tot)} seeds): d gas {dg.mean():+.7f} d eui {de.mean():+.7f} TOTAL mean "
          f"{tot.mean():+.8f} sd {tot.std():.8f} P(<0) {np.mean(tot < 0):.3f}; beta range [{bet.min():+.3f}, "
          f"{bet.max():+.3f}]")
# bootstrap of the 5-fold (seed 7) result over train rows, rescaled to private size (128,791 rows)
fold = np.random.default_rng(7).permutation(n) % 5
g2, e2 = pf_g.copy(), pf_e.copy()
for f in range(5):
    a, b = fold != f, fold == f
    g2[b], e2[b] = apply(pf_g[b], pf_e[b], A[b], beta_fit(pf_g[a], yg[a] - pf_g[a]))
sg0, sg1, se0, se1 = (yg - pf_g) ** 2, (yg - g2) ** 2, (ye - pf_e) ** 2, (ye - e2) ** 2
rng = np.random.default_rng(3)
bs = []
for _ in range(1000):
    i = rng.integers(0, n, 128791)
    bs.append(((np.sqrt(sg1[i].mean()) - np.sqrt(sg0[i].mean())) / SIG["gas"]
               + (np.sqrt(se1[i].mean()) - np.sqrt(se0[i].mean())) / SIG["eui"]) / 5)
bs = np.array(bs)
print(f"private-size bootstrap of the full-OOF 5-fold result: mean {bs.mean():+.8f} sd {bs.std():.8f} "
      f"P(<0) {np.mean(bs < 0):.3f}")
