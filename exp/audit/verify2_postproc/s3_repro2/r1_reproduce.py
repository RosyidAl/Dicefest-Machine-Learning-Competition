"""Independent reproduction (verify2_postproc, second pass) of claim S3: truncated-normal soft cap of gas / EUI
predictions near the hard gas truncation (train gas max 999,764.9).

Inputs: exp/audit/chain_H.npz only (honest holdout-chain predictions on H, P-side teacher predictions/residuals).
Own code, written from the claim text (not reusing exp/audit/postproc/*).

Honest evaluations of the S3 increment (after S1):
  a) k cross-fitted on H with many KFold(5) seeds (claimant used one split) and repeated split-halves,
  b) k fitted on the disjoint P side (P teacher OOF, no correction) and scored on all of H (fully out-of-sample),
  c) k = 1 (pure theory, no target fit at all),
  d) claimed constants (k fitted on all H) as an in-sample reference only.
Totals are per-target NRMSE change / 5.
"""
import json

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

OUT = "exp/audit/verify2_postproc/s3_repro2"
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
CLAIM = {"gas": dict(k=0.5046, a0=0.00806, b0=0.39297), "eui": dict(k=0.6574, a0=0.00839, b0=0.38715)}
T_CLAIM = 999764.8988

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AH, MH, AP, MP = A[H], M[H], A[P], M[P]
log = open(f"{OUT}/r1_reproduce.log", "w")
res = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


def nrmse(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def mills(b):
    """phi(b)/Phi(b), stable for very negative b"""
    return np.exp(norm.logpdf(b) - norm.logcdf(b))


def adj_tn(p, Tm, Mr, a0, b0):
    s = p * np.sqrt(a0 + b0 / Mr)
    b = (Tm - p) / s
    return -s * mills(b), b, s


# ---------------- 0. basic facts ----------------
yPg = z["gas_pP"] + z["gas_rP"]
yHg = z["gas_y"]
T_all = float(max(yPg.max(), yHg.max()))
T_P = float(yPg.max())
say(f"gas max: P {T_P:.4f}  H {yHg.max():.4f}  all {T_all:.4f} (claim T {T_CLAIM})")
say(f"gas rows in [990k,1e6]: P {(yPg >= 990000).sum()} H {(yHg >= 990000).sum()};  > 999765: {(yPg > 999765).sum() + (yHg > 999765).sum()}")
# eui identity check eui = gas / A on H and P
yPe = z["eui_pP"] + z["eui_rP"]
say(f"eui == gas/A: max rel err H {np.max(np.abs(z['eui_y'] - yHg / AH) / z['eui_y']):.2e}  P {np.max(np.abs(yPe - yPg / AP) / yPe):.2e}")
say(f"eui > T/A on train: H {(z['eui_y'] > T_all / AH + 1e-9).sum()} P {(yPe > T_all / AP + 1e-9).sum()}")
for m in ("gas", "eui"):
    say(f"base {m}: teacher {nrmse(z[m + '_y'], z[m + '_teacher'], m):.6f} corrected {nrmse(z[m + '_y'], z[m + '_corrected'], m):.6f}")

# ---------------- 1. noise model a0 + b0/M fitted on the P side ----------------
sd = {}
for m in ("gas", "eui"):
    rel2 = (z[m + "_rP"] / z[m + "_pP"]) ** 2
    X = np.column_stack([np.ones(len(P)), 1.0 / MP])
    a0, b0 = np.linalg.lstsq(X, rel2, rcond=None)[0]
    sd[m] = (float(a0), float(b0))
    say(f"P-side OLS (r/p)^2 ~ a0 + b0/M  {m}: a0 {a0:.5f} b0 {b0:.5f}   (claim {CLAIM[m]['a0']} / {CLAIM[m]['b0']})")
res["sd_fit_P"] = sd

# ---------------- 2. S1 (small-meter multiplicative shift, deltas from P side) ----------------
groups = {"M<=3": (MP <= 3, MH <= 3), "M=4": (MP == 4, MH == 4)}
pS1 = {}
for m in ("gas", "eui", "elec"):
    rP, pP = z[m + "_rP"], z[m + "_pP"]
    p = z[m + "_corrected"].copy()
    ds = {}
    for g, (gp, gh) in groups.items():
        d_sum = rP[gp].sum() / pP[gp].sum()
        d_mean = np.mean(rP[gp] / pP[gp])
        ds[g] = (round(float(d_sum), 5), round(float(d_mean), 5))
        p[gh] *= 1 + d_sum
    pS1[m] = p
    say(f"S1 {m} deltas (sum r/sum p, mean r/p): {ds}  H nrmse corrected {nrmse(z[m + '_y'], z[m + '_corrected'], m):.6f} -> S1 {nrmse(z[m + '_y'], p, m):.6f}")

# ---------------- 3. bias by distance to the cap (claim table) ----------------
bins = [(-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)]
for m in ("gas", "eui"):
    y = z[m + "_y"]
    for lab, p in (("corrected", z[m + "_corrected"]), ("afterS1", pS1[m])):
        Tm = T_CLAIM if m == "gas" else T_CLAIM / AH
        a, b, s = adj_tn(p, Tm, MH, CLAIM[m]["a0"], CLAIM[m]["b0"])
        r = y - p
        say(f"== {m} {lab}: residual by b=(T-p)/s bin (claim a0/b0)")
        for lo, hi in bins:
            sel = (b >= lo) & (b < hi)
            n = int(sel.sum())
            if n < 2:
                say(f"   [{lo},{hi}) n {n}")
                continue
            say(f"   [{lo},{hi}) n {n:6d} mean resid {r[sel].mean():+12.3f} se {r[sel].std(ddof=1) / np.sqrt(n):10.3f} "
                f"t {r[sel].mean() / (r[sel].std(ddof=1) / np.sqrt(n)):+5.2f}  mean adj(k=1) {a[sel].mean():+12.3f}")
    q = np.quantile(z[m + "_corrected"], 0.999)
    sel = z[m + "_corrected"] >= q
    r = y - z[m + "_corrected"]
    say(f"   top 0.1% predictions ({m}, corrected): n {sel.sum()} mean resid {r[sel].mean():+.3f} t {r[sel].mean() / (r[sel].std(ddof=1) / np.sqrt(sel.sum())):+.2f}")

# ---------------- 4. S3 honest increment over S1 ----------------
tot = {}
for m in ("gas", "eui"):
    y = z[m + "_y"]
    p = pS1[m]
    base = nrmse(y, p, m)
    a0, b0 = sd[m]
    Tm = T_P if m == "gas" else T_P / AH  # strictly honest cap: P-side max only
    adj, b, s = adj_tn(p, Tm, MH, a0, b0)
    r = y - p
    out = {"base_afterS1": base}
    k_all = float(adj @ r / (adj @ adj))
    out["k_inH_fullfit"] = k_all
    # (d) claimed constants (in-sample reference)
    adjc, _, _ = adj_tn(p, T_CLAIM if m == "gas" else T_CLAIM / AH, MH, CLAIM[m]["a0"], CLAIM[m]["b0"])
    out["claimed_consts_inH"] = nrmse(y, p + CLAIM[m]["k"] * adjc, m) - base
    out["k_inH_claimed_adj"] = float(adjc @ r / (adjc @ adjc))
    # (c) k = 1
    out["k1"] = nrmse(y, p + adj, m) - base
    # (a1) repeated KFold(5) cross-fitting with 200 seeds
    d5, ks = [], []
    for seed in range(1000, 1200):
        pn = p.copy()
        for tr_i, te_i in KFold(5, shuffle=True, random_state=seed).split(p):
            k = adj[tr_i] @ r[tr_i] / (adj[tr_i] @ adj[tr_i])
            ks.append(k)
            pn[te_i] = p[te_i] + k * adj[te_i]
        d5.append(nrmse(y, pn, m) - base)
    d5 = np.array(d5)
    out["cf5_mean"], out["cf5_sd_over_seeds"], out["cf5_min"], out["cf5_max"] = float(d5.mean()), float(d5.std()), float(d5.min()), float(d5.max())
    out["cf5_k_range"] = [float(np.min(ks)), float(np.max(ks))]
    # (a2) repeated split-half: fit k on one half, score the other half (both directions), 500 seeds
    rng = np.random.default_rng(20261020)
    dh = []
    for _ in range(500):
        perm = rng.permutation(len(p))
        h1, h2 = perm[: len(p) // 2], perm[len(p) // 2:]
        for fi, ev in ((h1, h2), (h2, h1)):
            k = adj[fi] @ r[fi] / (adj[fi] @ adj[fi])
            dh.append(nrmse(y[ev], p[ev] + k * adj[ev], m) - nrmse(y[ev], p[ev], m))
    dh = np.array(dh)
    out["half_mean"], out["half_sd"], out["half_frac_neg"] = float(dh.mean()), float(dh.std()), float((dh < 0).mean())
    # (b) k fitted on the P side (P teacher OOF, no correction, a0/b0 from P, cap from P), scored on all of H
    pP, rP = z[m + "_pP"], z[m + "_rP"]
    TmP = T_P if m == "gas" else T_P / AP
    adjP, bP, _ = adj_tn(pP, TmP, MP, a0, b0)
    kP = float(adjP @ rP / (adjP @ adjP))
    out["k_P"] = kP
    out["fitP_scoreH"] = nrmse(y, p + kP * adj, m) - base
    # P-side bias table (independent rows)
    for lo, hi in bins[:4]:
        sel = (bP >= lo) & (bP < hi)
        n = int(sel.sum())
        if n > 1:
            say(f"   P-side {m} [{lo},{hi}) n {n} mean resid {rP[sel].mean():+.3f} t {rP[sel].mean() / (rP[sel].std(ddof=1) / np.sqrt(n)):+.2f}")
    # per-row concentration and a delta-method se for the claimed-constant change
    dl = (y - p - CLAIM[m]["k"] * adjc) ** 2 - (y - p) ** 2
    rm = np.sqrt(np.mean((y - p) ** 2))
    out["claimed_se_rowwise"] = float(dl.std(ddof=1) / np.sqrt(len(dl)) / (2 * rm) / SIG[m])
    srt = np.sort(dl)
    out["rows_changed_gt_0.01sig"] = int((np.abs(CLAIM[m]["k"] * adjc) > 0.01 * SIG[m]).sum())
    out["loss_share_top10_rows"] = float(srt[:10].sum() / dl.sum())
    out["n_rows_worse"] = int((dl > 0).sum())
    out["n_rows_better"] = int((dl < 0).sum())
    res[m] = out
    say(f"== S3 {m}: base after S1 {base:.6f}  k in-H {k_all:.4f} (claim {CLAIM[m]['k']}; with claim adj {out['k_inH_claimed_adj']:.4f}); k from P {kP:.4f}")
    for key in ("claimed_consts_inH", "k1", "cf5_mean", "half_mean", "fitP_scoreH"):
        say(f"   {key:20s} d {out[key]:+.7f}  total {out[key] / 5:+.8f}")
    say(f"   cf5 over 200 seeds: sd {out['cf5_sd_over_seeds']:.7f} range [{out['cf5_min']:+.7f},{out['cf5_max']:+.7f}] k range {out['cf5_k_range']}")
    say(f"   split-half 1000 evals: sd {out['half_sd']:.7f} frac<0 {out['half_frac_neg']:.3f}")
    say(f"   rowwise se of claimed change (k fixed) {out['claimed_se_rowwise']:.7f}; rows changed >0.01 sigma {out['rows_changed_gt_0.01sig']} "
        f"({1e4 * out['rows_changed_gt_0.01sig'] / len(p):.1f} per 10k); better {out['n_rows_better']} worse {out['n_rows_worse']}; top-10-row share {out['loss_share_top10_rows']:.2f}")

for key in ("claimed_consts_inH", "k1", "cf5_mean", "half_mean", "fitP_scoreH"):
    tot[key] = (res["gas"][key] + res["eui"][key]) / 5
    say(f"TOTAL {key:20s} {tot[key]:+.8f}")
res["total"] = tot
json.dump(res, open(f"{OUT}/r1_reproduce.json", "w"), indent=1)
log.close()
