"""verify2_postproc second pass, S3 soft cap: (1) reverse-direction honest check (claimed k, fitted on H, scored on the
disjoint P-side teacher OOF), (2) test transfer of the change distribution (sub_v19vce = pre-S3 test predictions),
(3) check that the deployed post-processed files (sub_v19ncapp, sub_v19nca2pp) apply S3 as the claim states.
Reads only id/bld_dim_06/bld_dim_12 of test (usecols) and the submission CSVs."""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = "exp/audit/verify2_postproc/s3_repro2"
SIG = {"gas": 151750.04, "eui": 40.65}
COL = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm"}
CLAIM = {"gas": dict(k=0.5046, a0=0.00806, b0=0.39297), "eui": dict(k=0.6574, a0=0.00839, b0=0.38715)}
T = 999764.8988
log = open(f"{OUT}/r2_transfer.log", "w")
res = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


def s3_change(p, m, A, M):
    c = CLAIM[m]
    Tm = T if m == "gas" else T / A
    s = p * np.sqrt(c["a0"] + c["b0"] / M)
    b = (Tm - p) / s
    return -c["k"] * s * np.exp(norm.logpdf(b) - norm.logcdf(b))


def nrmse(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


# ---------- (1) reverse direction: claimed constants (fitted on H) scored on P teacher ----------
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
tot = 0.0
for m in ("gas", "eui"):
    pP, rP = z[m + "_pP"], z[m + "_rP"]
    y = pP + rP
    ch = s3_change(pP, m, A[P], M[P])
    d = nrmse(y, pP + ch, m) - nrmse(y, pP, m)
    dl = (y - pP - ch) ** 2 - (y - pP) ** 2
    se = dl.std(ddof=1) / np.sqrt(len(dl)) / (2 * np.sqrt(np.mean(rP ** 2))) / SIG[m]
    # P-side value of a hard clip at T only, for reference
    dclip = nrmse(y, np.minimum(pP, T if m == "gas" else T / A[P]), m) - nrmse(y, pP, m)
    tot += d / 5
    res[f"P_{m}"] = dict(d=d, se=float(se), hardclip=dclip, base=nrmse(y, pP, m))
    say(f"P-side teacher {m}: base {nrmse(y, pP, m):.6f}  claimed S3 d {d:+.7f} (rowwise se {se:.7f}, t {d / se:+.2f}) hard clip only d {dclip:+.7f}")
say(f"P-side TOTAL (claimed constants, k from H) {tot:+.8f}")
res["P_total"] = tot

# H change distribution (corrected predictions, no S1; S1 rows are far from the cap)
hstat = {}
for m in ("gas", "eui"):
    p = z[m + "_corrected"]
    ch = s3_change(p, m, A[H], M[H])
    hstat[m] = dict(gt001=1e4 * float(np.mean(np.abs(ch) > 0.01 * SIG[m])), gt01=1e4 * float(np.mean(np.abs(ch) > 0.1 * SIG[m])),
                    rms=float(np.sqrt(np.mean(ch ** 2)) / SIG[m]), maxabs=float(np.abs(ch).max() / SIG[m]))
    say(f"H {m}: per 10k rows |chg|>0.01sig {hstat[m]['gt001']:.1f}  >0.1sig {hstat[m]['gt01']:.1f}  rms {hstat[m]['rms']:.4f} sig  max {hstat[m]['maxabs']:.3f} sig;"
        f" pred > T(/A): {(p > (T if m == 'gas' else T / A[H])).sum()}")
res["H_change"] = hstat

# ---------- (2) test transfer ----------
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
subs = {n: pd.read_csv(f"submissions/{n}.csv") for n in ("sub_v19vce", "sub_v19nca", "sub_v19ncapp", "sub_v19nca2", "sub_v19nca2pp")}
for n, s in subs.items():
    assert (s["id"].values == te["id"].values).all(), n
tstat = {}
for m in ("gas", "eui"):
    p = subs["sub_v19vce"][COL[m]].values
    ch = s3_change(p, m, At, Mt)
    i = int(np.argmax(np.abs(ch)))
    tstat[m] = dict(gt001=1e4 * float(np.mean(np.abs(ch) > 0.01 * SIG[m])), gt01=1e4 * float(np.mean(np.abs(ch) > 0.1 * SIG[m])),
                    rms=float(np.sqrt(np.mean(ch ** 2)) / SIG[m]), maxabs=float(np.abs(ch).max() / SIG[m]),
                    max_row=dict(id=str(te["id"].values[i]), p=float(p[i]), new=float(p[i] + ch[i]), A=float(At[i]), M=float(Mt[i])),
                    n_over_cap=int((p > (T if m == "gas" else T / At)).sum()))
    say(f"test(v19vce) {m}: per 10k |chg|>0.01sig {tstat[m]['gt001']:.1f}  >0.1sig {tstat[m]['gt01']:.1f}  rms {tstat[m]['rms']:.4f} sig  "
        f"max {tstat[m]['maxabs']:.3f} sig  over cap {tstat[m]['n_over_cap']}  largest {tstat[m]['max_row']}")
    # distribution of distance to the cap: H vs test
    for lab, pp, AA, MM in (("H", z[m + "_corrected"], A[H], M[H]), ("test", p, At, Mt)):
        c = CLAIM[m]
        Tm = T if m == "gas" else T / AA
        b = (Tm - pp) / (pp * np.sqrt(c["a0"] + c["b0"] / MM))
        cnt = [1e4 * float(np.mean((b >= lo) & (b < hi))) for lo, hi in ((-9, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 3))]
        say(f"   {lab:4s} rows per 10k by b bin [<0, 0-.5, .5-1, 1-2, 2-3]: {np.round(cnt, 2).tolist()}")
res["test_change"] = tstat

# ---------- (3) deployed files: is S3 applied as stated? (rows with M > 4 are untouched by S1) ----------
for pre, post in (("sub_v19nca", "sub_v19ncapp"), ("sub_v19nca2", "sub_v19nca2pp")):
    for m in ("gas", "eui"):
        p0, p1 = subs[pre][COL[m]].values, subs[post][COL[m]].values
        ch = s3_change(p0, m, At, Mt)
        sel = Mt > 4
        diff = p1 - p0
        err = diff[sel] - ch[sel]
        big = np.abs(ch) > 1e-6 * SIG[m]
        notapplied = sel & big & (np.abs(diff) < 1e-9 * SIG[m])
        say(f"deployed {pre}->{post} {m}: rows M>4 max |actual - predicted S3| {np.abs(err).max() / SIG[m]:.2e} sig; "
            f"rows with S3 change but untouched {int(notapplied.sum())}; max p1 {p1.max():.1f}; rows over cap after {(p1 > (T if m == 'gas' else T / At) + 1e-6).sum()}")
        res[f"deployed_{post}_{m}"] = dict(max_err_sig=float(np.abs(err).max() / SIG[m]), untouched=int(notapplied.sum()))
json.dump(res, open(f"{OUT}/r2_transfer.json", "w"), indent=1)
log.close()
