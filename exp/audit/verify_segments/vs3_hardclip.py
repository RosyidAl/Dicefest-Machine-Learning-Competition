"""Adversarial re-derivation (leakage/optimism lens) of the claim:
  '7 test gas predictions in sub_v19vce exceed the hard data maximum of 1e6; a hard clip is worth about -0.000003 total'.

Independent of exp/audit/segments/seg_bounds.py and exp/audit/verify_segments/v1_clip_verify.py.
Light inputs only: chain_H.npz, parquet column subsets, submission columns, one raw-CSV column.
Every bound is either the constant 1e6 or taken from P rows only; scoring is on H (or on P-OOF with a constant bound).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/vs3_hardclip.py \
         > exp/audit/verify_segments/vs3_hardclip.log
"""
import itertools
import json

import numpy as np
import pandas as pd

OUT = "exp/audit/verify_segments/vs3_hardclip.json"
SIG = {"gas": 151750.04, "eui": 40.65}
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
CAP = 1e6
rng = np.random.default_rng(20261020)
res = {}


def nrmse(e, m, w=None):
    if w is None:
        return float(np.sqrt(np.mean(e ** 2)) / SIG[m])
    return float(np.sqrt(np.sum(w * e ** 2) / np.sum(w)) / SIG[m])


def boot_dtotal(y, p0, p1, m, reps=1000, chunk=50):
    """Poisson bootstrap over rows of the total-score change of replacing p0 by p1."""
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    out = []
    for _ in range(reps // chunk):
        w = rng.poisson(1.0, size=(chunk, y.size)).astype(np.float64)
        n = w.sum(1)
        d = (np.sqrt(w @ e1 / n) - np.sqrt(w @ e0 / n)) / SIG[m] / 5
        out.append(d)
    out = np.concatenate(out)
    return float(out.mean()), float(out.std()), float(np.mean(out < 0))


# ---------------------------------------------------------------- 1. data facts
tr = pd.read_parquet("cache/train.parquet", columns=["id", "bld_dim_06", GAS, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06"])
g_tr = tr[GAS].to_numpy(float)
A_tr = tr["bld_dim_06"].to_numpy(float)
raw_gas = pd.read_csv("data/raw/train.csv", usecols=[GAS])[GAS].to_numpy(float)
n_tr, n_te = len(tr), len(te)
print("=== 1. data facts")
print(f"train rows {n_tr}, test rows {n_te}, total {n_tr + n_te}; 3*n_tr == 7*n_te: {3 * n_tr == 7 * n_te}")
print(f"raw train.csv gas: rows {raw_gas.size}, max {raw_gas.max():.3f}, rows >= 1e6 {int((raw_gas >= CAP).sum())}; "
      f"cache == raw: {bool(np.allclose(np.sort(raw_gas), np.sort(g_tr), rtol=0, atol=1e-6))}")
top = np.sort(g_tr)[::-1]
print("top 5 train gas:", " ".join(f"{v:.1f}" for v in top[:5]))
rel = np.abs(tr[EUI].to_numpy(float) * A_tr - g_tr) / g_tr
print(f"identity eui*A == gas: max rel err {rel.max():.2e}")
# truncated-exponential MLE of the tail on [a, CAP): how many rows would lie above CAP without truncation?
a = 7e5
x = g_tr[(g_tr >= a) & (g_tr < CAP)] - a
L = CAP - a
lams = np.geomspace(1e-7, 1e-4, 4000)
ll = x.size * np.log(lams) - lams * x.sum() - x.size * np.log1p(-np.exp(-lams * L))
lam = lams[np.argmax(ll)]
exp_above = x.size * np.exp(-lam * L) / (1 - np.exp(-lam * L))
print(f"tail [7e5, 1e6): n {x.size}, MLE e-fold {1 / lam:.0f} kWh, expected rows >= 1e6 if untruncated {exp_above:.0f} (observed 0)")
# if only train had been filtered after an exact 70/30 split, train would be short by ~exp_above rows
print(f"if the 1e6 filter hit train only after a 70/30 split: expected train shortfall ~{exp_above:.0f} rows "
      f"(sd ~{np.sqrt(exp_above):.0f}); observed n_tr - 0.7*(n_tr+n_te) = {n_tr - 0.7 * (n_tr + n_te):.2f}")
res["facts"] = dict(n_tr=n_tr, n_te=n_te, exact_7_3=bool(3 * n_tr == 7 * n_te), gas_max=float(g_tr.max()),
                    n_ge_cap=int((g_tr >= CAP).sum()), efold=float(1 / lam), exp_above=float(exp_above))

# ---------------------------------------------------------------- 2. holdout chain (honest)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
assert np.array_equal(A, A_tr), "chain A differs from cache bld_dim_06"
assert np.allclose(z["gas_y"], g_tr[H]) and np.allclose(z["eui_y"], tr[EUI].to_numpy(float)[H])
assert np.intersect1d(P, H).size == 0 and P.size + H.size == n_tr
yP = g_tr[P]
maxP = float(yP.max())
print(f"\n=== 2. holdout chain: n P {P.size}, n H {H.size}; max gas on P {maxP:.1f}, on H {g_tr[H].max():.1f}")
print(f"gas_rP == y_P - gas_pP: {bool(np.allclose(z['gas_rP'], yP - z['gas_pP']))}")
res["H"] = {}
yH, AH = z["gas_y"], A[H]
for view in ("corrected", "teacher"):
    p0 = z[f"gas_{view}"]
    for name, cap in (("1e6", CAP), ("max_P", maxP)):
        p1 = np.minimum(p0, cap)
        ch = p1 != p0
        d = (nrmse(yH - p1, "gas") - nrmse(yH - p0, "gas"))
        bm, bs, bp = boot_dtotal(yH, p0, p1, "gas")
        improved = int(np.sum(np.abs(yH - p1)[ch] < np.abs(yH - p0)[ch]))
        print(f"  gas H {view:9s} clip {name:5s}: base {nrmse(yH - p0, 'gas'):.6f} rows {int(ch.sum()):3d} improved {improved:3d} "
              f"dNRMSE {d:+.8f} dTotal {d / 5:+.9f} (boot mean {bm:+.9f} sd {bs:.9f} P<0 {bp:.3f})")
        res["H"][f"gas_{view}_{name}"] = dict(rows=int(ch.sum()), improved=improved, dNRMSE=d, dTotal=d / 5, boot_sd=bs)
    if view == "corrected":
        sel = p0 > CAP
        print(f"    H corrected rows with pred > 1e6: n {int(sel.sum())}; overshoot p-1e6 sum {np.sum(p0[sel] - CAP):.0f}; "
              f"gap 1e6-y mean {np.mean(CAP - yH[sel]):.0f} median {np.median(CAP - yH[sel]):.0f} min {np.min(CAP - yH[sel]):.0f}")
yE = z["eui_y"]
for view in ("corrected", "teacher"):
    p0 = z[f"eui_{view}"]
    for name, cap in (("1e6/A", CAP / AH), ("max_P/A", maxP / AH)):
        p1 = np.minimum(p0, cap)
        ch = p1 != p0
        d = nrmse(yE - p1, "eui") - nrmse(yE - p0, "eui")
        bm, bs, bp = boot_dtotal(yE, p0, p1, "eui")
        print(f"  eui H {view:9s} clip {name:7s}: base {nrmse(yE - p0, 'eui'):.6f} rows {int(ch.sum()):3d} "
              f"dNRMSE {d:+.8f} dTotal {d / 5:+.9f} (boot sd {bs:.9f} P<0 {bp:.3f})")
        res["H"][f"eui_{view}_{name}"] = dict(rows=int(ch.sum()), dNRMSE=d, dTotal=d / 5, boot_sd=bs)
# P-side OOF view (P rows, predictions from P-internal folds; constant bound -> no fitting at all)
for m, y, cap in (("gas", yP, CAP), ("eui", tr[EUI].to_numpy(float)[P], CAP / A[P])):
    p0 = z[f"{m}_pP"]
    p1 = np.minimum(p0, cap)
    d = nrmse(y - p1, m) - nrmse(y - p0, m)
    print(f"  {m} P-OOF teacher clip 1e6{'/A' if m == 'eui' else ''}: base {nrmse(y - p0, m):.6f} rows {int((p1 != p0).sum()):3d} "
          f"dNRMSE {d:+.8f} dTotal {d / 5:+.9f}  (rows scaled to H size: dTotal x {P.size / H.size:.2f} density-neutral)")
    res["H"][f"{m}_Poof_1e6"] = dict(rows=int((p1 != p0).sum()), dTotal=d / 5)

# ---------------------------------------------------------------- 3. the submission (test)
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["id", GAS, EUI])
assert np.array_equal(sub["id"].to_numpy(), te["id"].to_numpy()), "id order differs from cache test"
pg = sub[GAS].to_numpy(float)
pe = sub[EUI].to_numpy(float)
At = te["bld_dim_06"].to_numpy(float)
og = np.where(pg > CAP)[0]
oe = np.where(pe * At > CAP)[0]
print(f"\n=== 3. sub_v19vce test: gas rows > 1e6: {og.size}: " + " ".join(f"{v:.0f}" for v in np.sort(pg[og])[::-1]))
print(f"    rows > train max {g_tr.max():.0f}: {int((pg > g_tr.max()).sum())}")
print(f"    EUI rows with eui*A > 1e6: {oe.size}; overlap with gas rows {np.intersect1d(og, oe).size}; "
      f"eui overshoot (eui - 1e6/A): " + " ".join(f"{v:.2f}" for v in np.sort(pe[oe] - CAP / At[oe])[::-1]))
print(f"    gas/eui consistency on those rows: max |eui*A/gas - 1| {np.max(np.abs(pe[og] * At[og] / pg[og] - 1)):.4f}")
res["test"] = dict(gas_rows=int(og.size), gas_vals=sorted(pg[og].tolist(), reverse=True), eui_rows=int(oe.size))

# reference conditional gap distribution: rows whose prediction exceeds the cap, H corrected + P-OOF teacher
ref_gap = {"H corrected": CAP - yH[z["gas_corrected"] > CAP], "P-OOF teacher": CAP - yP[z["gas_pP"] > CAP]}
ref_gap["pooled"] = np.concatenate(list(ref_gap.values()))
base_nr = {"gas": nrmse(yH - z["gas_corrected"], "gas"), "eui": nrmse(yE - z["eui_corrected"], "eui")}


def test_dtotal(dsse, m, n):
    """total-score change for a test subset of n rows given its SSE change, base NRMSE assumed = H corrected."""
    mse0 = (base_nr[m] * SIG[m]) ** 2
    return (np.sqrt(mse0 + dsse / n) - np.sqrt(mse0)) / SIG[m] / 5


ov_g = pg[og] - CAP
ov_e = pe[oe] - CAP / At[oe]
print("\n=== 4. transfer to test (base NRMSE taken from H corrected)")
for nm, g in ref_gap.items():
    print(f"  ref {nm:14s}: n {g.size:3d} mean gap {g.mean():8.0f} median {np.median(g):8.0f} min {g.min():8.0f}")
rows = []
for nm, g in [("guaranteed (gap 0)", np.zeros(1))] + list(ref_gap.items()):
    gm = float(g.mean())
    d_g = float(np.sum(ov_g ** 2 + 2 * ov_g * gm))
    d_e = float(np.sum(ov_e ** 2 + 2 * ov_e * gm / At[oe]))
    tg, tE = test_dtotal(-d_g, "gas", n_te), test_dtotal(-d_e, "eui", n_te)
    # bootstrap sd over the reference rows (uncertainty of the conditional mean gap)
    if g.size > 1:
        gb = rng.choice(g, size=(4000, g.size)).mean(1)
        tb = np.array([test_dtotal(-np.sum(ov_g ** 2 + 2 * ov_g * v), "gas", n_te) for v in gb])
        sd = float(tb.std())
    else:
        sd = 0.0
    print(f"  {nm:20s}: gas dTotal {tg:+.9f} (sd {sd:.9f})  eui dTotal {tE:+.9f}  sum {tg + tE:+.9f}")
    rows.append(dict(ref=nm, mean_gap=gm, gas=tg, eui=tE, sum=tg + tE, gas_sd=sd))
res["transfer"] = rows

# public (30%) / private (70%) split of the 7 gas rows: exact enumeration over assignments
gm = float(ref_gap["pooled"].mean())
dsse_i = ov_g ** 2 + 2 * ov_g * gm
n_pub, n_priv = round(0.3 * n_te), n_te - round(0.3 * n_te)
vals_pub, vals_priv, probs = [], [], []
for mask in itertools.product((0, 1), repeat=og.size):  # 1 = private
    mk = np.array(mask, bool)
    probs.append(0.7 ** mk.sum() * 0.3 ** (~mk).sum())
    vals_priv.append(test_dtotal(-dsse_i[mk].sum(), "gas", n_priv))
    vals_pub.append(test_dtotal(-dsse_i[~mk].sum(), "gas", n_pub))
probs, vals_pub, vals_priv = map(np.array, (probs, vals_pub, vals_priv))


def wq(v, p, q):
    o = np.argsort(v)
    c = np.cumsum(p[o])
    return float(v[o][np.searchsorted(c, q)])


print(f"  gas private LB (n {n_priv}): mean {np.sum(probs * vals_priv):+.9f}  P10 {wq(vals_priv, probs, .1):+.9f}  P90 {wq(vals_priv, probs, .9):+.9f}")
print(f"  gas public  LB (n {n_pub}): mean {np.sum(probs * vals_pub):+.9f}  P10 {wq(vals_pub, probs, .1):+.9f}  P90 {wq(vals_pub, probs, .9):+.9f}")
res["lb_split"] = dict(priv_mean=float(np.sum(probs * vals_priv)), pub_mean=float(np.sum(probs * vals_pub)))

# rate comparison: rows per 100k with pred above t (filter-before-split check; test would show an excess if untruncated)
print("\n=== 5. rows per 100k with gas pred above t: test sub | H corrected | P-OOF teacher")
for t in (8e5, 9e5, 9.5e5, 1e6):
    rt, rh, rp = (np.mean(v > t) * 1e5 for v in (pg, z["gas_corrected"], z["gas_pP"]))
    print(f"  t {t:9.0f}: {rt:7.1f} {rh:7.1f} {rp:7.1f}")
sh = np.mean(z["gas_pP"][(yP >= 9.5e5)] > 9e5)
exp_te_untrunc = exp_above * n_te / n_tr
null9 = np.mean(np.concatenate([z["gas_corrected"], z["gas_pP"]]) > 9e5) * n_te
obs9 = int(np.sum(pg > 9e5))
print(f"  P rows with truth >= 9.5e5: share with pred > 9e5 = {sh:.3f}; if test were untruncated it would hold ~{exp_te_untrunc:.0f} rows "
      f">= 1e6 -> >= {exp_te_untrunc * sh:.0f} extra rows with pred > 9e5; observed {obs9}, train-side rate implies {null9:.0f}")

# ---------------------------------------------------------------- 6. no double counting / already deployed?
for f in ("exp/audit/segments/sub_v19vce_cap_k0.08.csv", "submissions/sub_v19nca2pp.csv", "submissions/BEST_CURRENT.csv"):
    s = pd.read_csv(f, usecols=["id", GAS, EUI])
    assert np.array_equal(s["id"].to_numpy(), te["id"].to_numpy())
    g, e = s[GAS].to_numpy(float), s[EUI].to_numpy(float)
    print(f"  {f}: gas max {g.max():.0f}, rows > 1e6 {int((g > CAP).sum())}, rows eui*A > 1e6 {int((e * At > CAP * (1 + 1e-12)).sum())}")
json.dump(res, open(OUT, "w"), indent=1)
