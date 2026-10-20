"""Analysis of claim S1 on the full train OOF (own VC refit from s02_fullvc.npz) and on the test side.

1. multiplicative LS bias by M group on the full in-sample VC fit and on the nested VC fit
2. the action's deltas (sum(pf r)/sum(pf^2), in-sample full fit) vs the claimed constants
3. honest gains on all train rows:
   a. fully nested: VC blend AND deltas fitted on 4/5 of the rows, applied to the other 1/5 (mirrors deployment)
   b. deltas cross-fitted (5-fold, 50 seeds) on the in-sample full fit
   c. robustness: gain without the k most influential rows (EUI), and per-group split
4. test side: rows touched, correction size on small-M test rows (sub_v19vce minus own VC test blend), level of test
   predictions vs train OOF predictions inside the groups, analytic expected test gain
5. the claim's candidate file exp/audit/postproc/sub_v19vce_pp.csv: ratio to sub_v19vce on the touched rows

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/r2_smallM/s03_analysis.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from dup_fix import KEY, duplicate_targets  # noqa: E402

OUT = "exp/audit/verify2_postproc/r2_smallM"
GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
COL = dict(gas=GAS, eui=EUI, elec=EL)
SIG = dict(gas=151750.04, eui=40.65, elec=41725.37)
MET = ("gas", "eui", "elec")
CLAIM = dict(g23=dict(gas=0.19942, eui=0.18166, elec=0.18714), g4=dict(gas=0.06202, eui=0.03016, elec=0.07098))
GROUPS = ("g23", "g4")
res = {}

z = np.load(f"{OUT}/s02_fullvc.npz")
tr = pd.read_parquet("cache/train.parquet", columns=sorted(set(KEY + [GAS, EL, EUI, "target_clean_building_ratio",
                                                                       "target_electric_heating_adoption"])))
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
fold = z["fold"]


def grp(Mv):
    g = np.full(len(Mv), "", dtype=object)
    g[(Mv == 2) | (Mv == 3)] = "g23"
    g[Mv == 4] = "g4"
    return g


G, Gt = grp(M), grp(Mt)


def ls_bias(p, r):
    s2 = float(np.sum(p * p))
    d = float(np.sum(p * r)) / s2
    return d, float(np.sqrt(np.sum(p * p * (r - d * p) ** 2))) / s2


def deltas(p, r, g):
    return {k: ls_bias(p[g == k], r[g == k])[0] for k in GROUPS}


def apply(p, g, dl):
    q = p.copy()
    for k in GROUPS:
        q[g == k] *= 1.0 + dl[k]
    return q


def dn(y, p0, p1, sig):
    return float((np.sqrt(np.mean((y - p1) ** 2)) - np.sqrt(np.mean((y - p0) ** 2))) / sig)


def boot_sd(y, p0, p1, sig, n_out=None, nboot=1000, seed=0):
    """sd of the NRMSE change for a scored sample of n_out rows drawn from these rows (deltas fixed)"""
    rng = np.random.default_rng(seed)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    n_out = n_out or len(y)
    v = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, len(y), n_out)
        v[b] = (np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())) / sig
    return float(v.std())


print("rows by M  train:", {int(m): int((M == m).sum()) for m in range(1, 6)},
      " test:", {int(m): int((Mt == m).sum()) for m in range(1, 6)})

# 1 + 2. bias tables and action deltas
print("\n1. LS multiplicative bias (se, t) on full train: in-sample VC fit | nested VC fit")
bias = {}
for m in MET:
    y = tr[COL[m]].values
    for lab, sel in [("M2-3", G == "g23"), ("M=4", G == "g4"), ("M=5", M == 5), ("M=6", M == 6),
                     ("M>=7", M >= 7)]:
        d1, s1 = ls_bias(z[f"pf_{m}"][sel], y[sel] - z[f"pf_{m}"][sel])
        d2, s2 = ls_bias(z[f"pn_{m}"][sel], y[sel] - z[f"pn_{m}"][sel])
        bias[f"{m}_{lab}"] = dict(n=int(sel.sum()), insample=(d1, s1), nested=(d2, s2))
        print(f"  {m:4s} {lab:5s} n {sel.sum():6d} | in-sample {d1:+.4f} ({s1:.4f}, t {d1 / s1:+.1f}) "
              f"| nested {d2:+.4f} ({s2:.4f}, t {d2 / s2:+.1f})")
res["bias"] = bias
print("\n2. action deltas from the in-sample full fit vs claimed constants")
act = {}
for m in MET:
    y = tr[COL[m]].values
    dl = deltas(z[f"pf_{m}"], y - z[f"pf_{m}"], G)
    act[m] = dl
    print(f"  {m:4s} d23 {dl['g23']:+.5f} (claim {CLAIM['g23'][m]:+.5f})  d4 {dl['g4']:+.5f} "
          f"(claim {CLAIM['g4'][m]:+.5f})")
res["action_deltas"] = act

# 3a. fully nested gain
print("\n3a. fully nested (VC blend and deltas fitted on 4/5, scored on 1/5), all train rows")
nest, tot, tot_sd2 = {}, 0.0, 0.0
for m in MET:
    y, pn = tr[COL[m]].values, z[f"pn_{m}"]
    p1 = pn.copy()
    dls = []
    for f in range(5):
        a, b = fold != f, fold == f
        pfa = z[f"pfit_{m}_{f}"]
        dl = deltas(pfa, y[a] - pfa, G[a])
        dls.append(dl)
        p1[b] = apply(pn[b], G[b], dl)
    g = dn(y, pn, p1, SIG[m])
    sd_all = boot_sd(y, pn, p1, SIG[m])
    sd_pub = boot_sd(y, pn, p1, SIG[m], n_out=55196, seed=1)
    sd_prv = boot_sd(y, pn, p1, SIG[m], n_out=128791, seed=2)
    part = {}
    for k in GROUPS:
        q = pn.copy()
        q[G == k] = p1[G == k]
        part[k] = dn(y, pn, q, SIG[m])
    nest[m] = dict(dnrmse=g, sd_train=sd_all, sd_public=sd_pub, sd_private=sd_prv, by_group=part,
                   deltas_folds=dls)
    tot += g / 5
    print(f"  {m:4s} dNRMSE {g:+.7f} (sd: train {sd_all:.7f}, public-size {sd_pub:.7f}, private-size {sd_prv:.7f}) "
          f"g23 {part['g23']:+.7f} g4 {part['g4']:+.7f} | fold d23 "
          + " ".join(f"{d['g23']:+.3f}" for d in dls))
print(f"  TOTAL change {tot:+.7f}")
res["3a_nested"] = dict(per_metric=nest, total=tot)

# 3b. cross-fitted deltas on the in-sample full fit
print("\n3b. deltas cross-fitted on the in-sample full fit (5-fold, 50 seeds)")
cf = {m: [] for m in MET}
for s in range(50):
    fo = np.random.default_rng(500 + s).integers(0, 5, len(M))
    for m in MET:
        y, pf = tr[COL[m]].values, z[f"pf_{m}"]
        p1 = pf.copy()
        for f in range(5):
            a, b = fo != f, fo == f
            p1[b] = apply(pf[b], G[b], deltas(pf[a], y[a] - pf[a], G[a]))
        cf[m].append(dn(y, pf, p1, SIG[m]))
tots = sum(np.array(cf[m]) for m in MET) / 5
print("  " + " ".join(f"{m} {np.mean(cf[m]):+.7f}" for m in MET)
      + f" | TOTAL {tots.mean():+.7f} (seed sd {tots.std():.7f}, frac<0 {np.mean(tots < 0):.2f})")
res["3b_crossfit_full"] = dict({m: float(np.mean(cf[m])) for m in MET}, total=float(tots.mean()),
                               seed_sd=float(tots.std()))

# 3c. influence: EUI gain concentration in the nested scheme
print("\n3c. concentration of the gain (nested scheme): share of the squared-error reduction from the top rows")
infl = {}
for m in MET:
    y, pn = tr[COL[m]].values, z[f"pn_{m}"]
    p1 = pn.copy()
    for f in range(5):
        a, b = fold != f, fold == f
        pfa = z[f"pfit_{m}_{f}"]
        p1[b] = apply(pn[b], G[b], deltas(pfa, y[a] - pfa, G[a]))
    contrib = (y - p1) ** 2 - (y - pn) ** 2  # negative = better
    idx = np.where(contrib != 0)[0]
    order = idx[np.argsort(contrib[idx])]
    tot_c = contrib.sum()
    top = {k: float(contrib[order[:k]].sum() / tot_c) for k in (1, 3, 5, 10)}
    worst = {k: float(contrib[order[::-1][:k]].sum() / tot_c) for k in (1, 3, 5)}
    n_better = int((contrib[idx] < 0).sum())
    trimmed = {}
    for k in (1, 3, 5):  # drop the k most helpful rows from the scored set
        keep = np.ones(len(y), bool)
        keep[order[:k]] = False
        trimmed[k] = dn(y[keep], pn[keep], p1[keep], SIG[m])
    infl[m] = dict(top_share=top, worst_share=worst, rows_touched=int(len(idx)), rows_better=n_better,
                   trimmed=trimmed)
    print(f"  {m:4s} touched {len(idx)}, better {n_better} | share of net gain from top 1/3/5/10 rows "
          + " ".join(f"{top[k]:.2f}" for k in (1, 3, 5, 10))
          + " | worst 1/3/5 rows " + " ".join(f"{worst[k]:+.2f}" for k in (1, 3, 5))
          + " | dNRMSE without top 1/3/5 helpful rows " + " ".join(f"{trimmed[k]:+.7f}" for k in (1, 3, 5)))
res["3c_influence"] = infl

# 4. test side
print("\n4. test side")
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
dup = duplicate_targets(tr, te)
hit = ~np.isnan(dup[:, 0])
print(f"  test rows: g23 {int((Gt == 'g23').sum())} (dup {int((hit & (Gt == 'g23')).sum())}), "
      f"g4 {int((Gt == 'g4').sum())} (dup {int((hit & (Gt == 'g4')).sum())}), M==1 {int((Mt == 1).sum())}, "
      f"dups {int(hit.sum())}")
tside = {}
for m in MET:
    pa, sv = z[f"pa_{m}"], sub[COL[m]].values
    corr = sv - np.clip(pa, 0, None)
    nd = ~hit
    row = {}
    for k in GROUPS:
        s_ = nd & (Gt == k)
        rc = float(np.sum(pa[s_] * corr[s_]) / np.sum(pa[s_] ** 2))
        trs = G == k
        y = tr[COL[m]].values
        lvl_te = float(np.mean(np.log(np.clip(pa[s_], 1e-9, None) / (At[s_] if m != "eui" else 1.0))))
        lvl_tr = float(np.mean(np.log(np.clip(z[f"pf_{m}"][trs], 1e-9, None) / (A[trs] if m != "eui" else 1.0))))
        lvl_y = float(np.mean(np.log(np.clip(y[trs], 1e-9, None) / (A[trs] if m != "eui" else 1.0))))
        row[k] = dict(rel_corr_test=rc, mean_log_level_test_pred=lvl_te, mean_log_level_train_oof=lvl_tr,
                      mean_log_level_train_y=lvl_y)
    others = nd & (Gt == "")
    rc_o = float(np.sum(pa[others] * corr[others]) / np.sum(pa[others] ** 2))
    row["rel_corr_test_other"] = rc_o
    row["corr_rms_over_sigma_all"] = float(np.sqrt(np.mean(corr[nd] ** 2)) / SIG[m])
    tside[m] = row
    print(f"  {m:4s} relative correction on test (sum pa*corr/sum pa^2): g23 {row['g23']['rel_corr_test']:+.4f} "
          f"g4 {row['g4']['rel_corr_test']:+.4f} other {rc_o:+.4f} | mean log level (per area for gas/elec): "
          f"g23 test-pred {row['g23']['mean_log_level_test_pred']:.3f} train-oof "
          f"{row['g23']['mean_log_level_train_oof']:.3f} train-y {row['g23']['mean_log_level_train_y']:.3f}; "
          f"g4 test-pred {row['g4']['mean_log_level_test_pred']:.3f} train-oof "
          f"{row['g4']['mean_log_level_train_oof']:.3f} train-y {row['g4']['mean_log_level_train_y']:.3f}")
res["test_side"] = tside

# 4b. analytic expected test gain: E[dMSE] = sum_test (d^2 - 2 d b) p^2 / n_test, b = true bias,
# with d = claimed constant and b taken from the nested train estimate; deltas' estimation noise included via se^2
print("\n4b. analytic expected test gain (sub_v19vce predictions, constants as claimed)")
tot_a = 0.0
for m in MET:
    sv = sub[COL[m]].values.astype(float)
    y = tr[COL[m]].values
    rmse0 = np.sqrt(np.mean((y - z[f"pn_{m}"]) ** 2))  # proxy for the test RMSE before the shift
    dmse = 0.0
    for k in GROUPS:
        s_ = (~hit) & (Gt == k)
        b, se = bias[f"{m}_{'M2-3' if k == 'g23' else 'M=4'}"]["nested"]
        d = CLAIM[k][m]
        dmse += float(np.sum((d * d - 2 * d * b) * sv[s_] ** 2)) / len(sv)
    g = (np.sqrt(rmse0 ** 2 + dmse) - rmse0) / SIG[m]
    tot_a += g / 5
    print(f"  {m:4s} expected dNRMSE {g:+.7f}")
print(f"  TOTAL expected change {tot_a:+.7f}")
res["4b_analytic_total"] = tot_a

# 5. candidate file check
print("\n5. claim candidate exp/audit/postproc/sub_v19vce_pp.csv vs sub_v19vce.csv (ratio on touched rows)")
cand = pd.read_csv("exp/audit/postproc/sub_v19vce_pp.csv", float_precision="round_trip")
assert (cand["id"].values == sub["id"].values).all()
cc = {}
for m in MET:
    a0, a1 = sub[COL[m]].values, cand[COL[m]].values
    ratio = np.where(a0 > 0, a1 / np.where(a0 > 0, a0, 1), np.nan)
    st = {}
    for lab, s_ in [("g23", (~hit) & (Gt == "g23")), ("g4", (~hit) & (Gt == "g4")), ("dup", hit), ("M1", Mt == 1),
                    ("other", (~hit) & (Gt == "") & (Mt != 1))]:
        changed = int(np.sum(a0[s_] != a1[s_]))
        st[lab] = dict(n=int(s_.sum()), changed=changed,
                       ratio_min=float(np.nanmin(ratio[s_])) if s_.sum() else None,
                       ratio_max=float(np.nanmax(ratio[s_])) if s_.sum() else None)
    cc[m] = st
    print(f"  {m:4s} " + " | ".join(f"{k}: n {v['n']} changed {v['changed']} ratio {v['ratio_min']:.5f}..{v['ratio_max']:.5f}"
                                    if v["n"] else f"{k}: n 0" for k, v in st.items()))
other_cols = [c for c in sub.columns if c not in ("id", GAS, EL, EUI)]
print("  other columns identical:", {c: bool((sub[c].values == cand[c].values).all()) for c in other_cols})
res["5_candidate"] = cc
json.dump(res, open(f"{OUT}/s03_analysis.json", "w"), indent=1, default=float)
print("\nwrote", f"{OUT}/s03_analysis.json")
