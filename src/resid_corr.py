"""Group residual correction of the blend.

Blend predictions are shifted by shrunken group means of the blend's OOF residuals over three spatial groups:
zone_region_id, climate cell and a +-WIN window of sub_n. The models already see target encodings of these
groups but still under-use them. The correction weights are fitted on the holdout of exp_hold.py: H rows play
the test rows and P-OOF residuals play the train OOF residuals, the same set-up as train OOF -> test. Weights
cannot be fitted on full-data OOF: OOF residuals of rows in other folds come from models that saw the row's
own target through the target encodings, which biases them.

usage: python src/resid_corr.py fit [--extra_gas a,b] [--extra_elec c] [--lam f]   holdout fit -> f (+ H gains)
       python src/resid_corr.py apply <tag> <new> [--lam f]   blend lists of exp/prune_<tag>.log -> sub_<new>.csv
--extra_*: tune-set holdout members (exp/hold/tune) added to the holdout blend, to match the full-data blend.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from blend import nnls_blend  # noqa: E402
from common import SIGMAS, TARGET_COLS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402

K = 30    # shrinkage n / (n + K) of a group mean
WIN = 30  # half-width of the sub_n window
GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
METRICS = {"gas": GAS, "eui": EUI, "elec": EL}
# holdout blend: every exp/hold member, with the depth-10 XGB swaps of sub_v8
X8 = {"x4_gpm": "xt_gpm_d10_lr03", "x4_gpgea": "xt_gpgea_d10_lr03", "x4_lgpm": "xt_lgpm_d10_lr03",
      "x4_gpn": "xt_gpn_d10_lr05"}


def group_keys(tr, te, subn, subnt):
    """(zone, climate cell, sub_n) codes for train and test rows"""
    both = pd.concat([tr, te])
    zone = pd.factorize(both["zone_region_id"])[0]
    cell = pd.factorize(pd.Series(list(zip(both["env_climate_01"], both["env_climate_03"], both["env_climate_05"]))))[0]
    n = len(tr)
    return (zone[:n], cell[:n], subn), (zone[n:], cell[n:], subnt)


def group_mean(gf, ga, num, den):
    """n/(n+K) * sum(num)/sum(den) over fit rows in the apply row's group"""
    ng = max(gf.max(), ga.max()) + 1
    s, dn, n = (np.bincount(gf, v, ng)[ga] for v in (num, den, np.ones(len(gf))))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def window_mean(sf, sa, num, den):
    """as group_mean, over fit rows with |sub_n - sub_n(apply row)| <= WIN"""
    m = max(sf.max(), sa.max()) + 1
    lo, hi = np.clip(sa - WIN, 0, m), np.clip(sa + WIN + 1, 0, m)
    s, dn, n = ((c := np.r_[0, np.cumsum(np.bincount(sf, v, m))])[hi] - c[lo] for v in (num, den, np.ones(len(sf))))
    return n / (n + K) * np.where(dn > 0, s / np.where(dn > 0, dn, 1), 0.0)


def features(metric, kf, ka, r, p_fit, p_app, A_fit, A_app):
    """Correction features of apply rows from fit-row residuals r on the metric's scale.
    gas: per-area shift times the row's area; eui: area-weighted mean residual (EUI noise variance ~ 1/area);
    elec: relative shift times the row's prediction."""
    if metric == "gas":
        num, den, mult = r, A_fit, A_app
    elif metric == "eui":
        num, den, mult = A_fit * r, A_fit, 1.0
    else:
        num, den, mult = r, p_fit, p_app
    return np.column_stack([group_mean(kf[0], ka[0], num, den), group_mean(kf[1], ka[1], num, den),
                            window_mean(kf[2], ka[2], num, den)]) * np.reshape(mult, (-1, 1))


def blend_preds(Pf, Pa, yf, metric, A_fit, A_app):
    """NNLS blend as in make_submission.py; returns predictions on the metric's scale for fit and apply rows"""
    if metric == "gas":
        w = nnls_blend(Pf, yf, A_fit)
        return Pf @ w * A_fit, Pa @ w * A_app
    w = nnls_blend(Pf, yf)
    return Pf @ w, Pa @ w


def fit(extra_gas, extra_elec, lam_file):
    from train_lib import Data
    d = Data()
    tr, te = load()
    P, H = holdout_split(len(tr))
    kt, _ = group_keys(tr, te, d.subn, d.subnt)
    kP, kH = tuple(k[P] for k in kt), tuple(k[H] for k in kt)
    is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
    mem = sorted(f[:-4] for f in os.listdir("exp/hold/pred") if os.path.getsize(f"exp/hold/pred/{f}") > 0)
    gas = [X8.get(m, m) for m in mem if not is_el(m)] + extra_gas
    el = [m for m in mem if is_el(m)] + extra_elec
    path = lambda n, part: next(f for f in (f"exp/hold/{part}/{n}.npy", f"exp/hold/tune/{part}/{n}.npy")  # noqa: E731
                                if os.path.exists(f))
    lam = {}
    for metric, col in METRICS.items():
        names = el if metric == "elec" else gas
        Po = np.column_stack([np.load(path(n, "oof")) for n in names])
        Ph = np.column_stack([np.load(path(n, "pred")) for n in names])
        yv = tr[EL if metric == "elec" else EUI].values
        A = tr["bld_dim_06"].values
        pP, pH = blend_preds(Po, Ph, yv[P], metric, A[P], A[H])
        scale = A if metric == "gas" else np.ones(len(A))
        rP, rH = scale[P] * yv[P] - pP, scale[H] * yv[H] - pH
        F = features(metric, kP, kH, rP, pP, pH, A[P], A[H])
        pc = np.zeros(len(H))
        for a, b in KFold(5, shuffle=True, random_state=0).split(F):
            pc[b] = F[b] @ np.linalg.lstsq(F[a], rH[a], rcond=None)[0]
        lam[metric] = np.linalg.lstsq(F, rH, rcond=None)[0].tolist()
        base = np.sqrt(np.mean(rH ** 2)) / SIGMAS[col]
        print(f"{metric:4s} H {base:.5f} -> cross-fitted {np.sqrt(np.mean((rH - pc) ** 2)) / SIGMAS[col]:.5f} "
              f"lam {np.round(lam[metric], 3)}", flush=True)
    json.dump(lam, open(lam_file, "w"), indent=1)
    print("wrote", lam_file)


def apply(tag, new, lam_file):
    from train_lib import Data
    d = Data()
    tr, te = load()
    kt, ke = group_keys(tr, te, d.subn, d.subnt)
    lam = json.load(open(lam_file))
    log = open(f"exp/prune_{tag}.log").read()
    runs = {k: re.search(rf"^{k} runs: (\S+)", log, re.M).group(1).split(",") for k in ("gas", "elec")}
    sub = pd.read_csv(f"submissions/sub_{tag}.csv")
    A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
    for metric, col in METRICS.items():
        names = runs["elec" if metric == "elec" else "gas"]
        Po = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
        Pt = np.column_stack([np.load(f"exp/pred/{n}.npy") for n in names])
        yv = tr[EL if metric == "elec" else EUI].values
        pf, pa = blend_preds(Po, Pt, yv, metric, A, At)
        pa_raw = np.clip(pa, 0, None)
        assert np.allclose(pa_raw, sub[col].values, rtol=1e-6, atol=1e-6), f"{col}: blend differs from sub_{tag}"
        scale = A if metric == "gas" else np.ones(len(A))
        r = scale * yv - pf
        corr = features(metric, kt, ke, r, pf, pa, A, At) @ np.array(lam[metric])
        sub[col] = np.clip(pa + corr, 0, None)
        print(f"{metric:4s} lam {np.round(lam[metric], 3)} correction rms/sigma {np.sqrt(np.mean(corr ** 2)) / SIGMAS[col]:.5f}",
              flush=True)
    assert np.isfinite(sub[TARGET_COLS].values).all()
    out = f"submissions/sub_{new}.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "apply"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--extra_gas", default="")
    ap.add_argument("--extra_elec", default="p1_lepm")
    ap.add_argument("--lam", default="exp/resid_lam.json")
    a = ap.parse_args()
    if a.mode == "fit":
        fit([m for m in a.extra_gas.split(",") if m], [m for m in a.extra_elec.split(",") if m], a.lam)
    else:
        apply(*a.tags, a.lam)
