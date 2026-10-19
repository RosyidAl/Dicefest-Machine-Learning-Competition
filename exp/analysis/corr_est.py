"""PP0007: alternative estimators for the similarity-kNN residual-correction features.

Each kNN feature is a shrunken ratio of sums over the k nearest fit rows: sum(num) / sum(den) * mult. The current
estimator per metric (src/resid_knn.py all_features) is
  gas : num = r, den = A, mult = A_app  (per-area shift)       eui : num = A r, den = A, mult = 1 (area-weighted mean)
  elec: num = r, den = p, mult = p_app  (relative shift)
Alternatives tested here, for every config of the select_corr grid:
  rel: num = r, den = p, mult = p_app (relative shift)        pl : num = r, den = 1, mult = 1 (plain mean)
  ar : num = r, den = A, mult = A_app (per-area shift)
The neighbour sets do not depend on the metric, so one neighbour search per config gives all metrics and estimators.
Holdout chain as exp/analysis/corr_honest.py (VC blend A2xM2xp2 fitted on P-OOF, applied to H). Honest split-half
check: greedy additions of alternative columns to the current configs (gas, elec: CONFIGS; eui:
exp/analysis/corr_final.json) chosen on one half of H, scored on the other half (cross-fitted lstsq there).

usage: python exp/analysis/corr_est.py            (writes the columns to exp/corrsel/est_*.npy, then the honest check)
       python exp/analysis/corr_est.py final gas,eui    (greedy on all of H for the listed metrics -> corr_est_final.json)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "exp/analysis")
from blend import blend_vc  # noqa: E402
from common import SIGMAS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402
from resid_corr import EL, EUI, METRICS, X8, features, group_keys  # noqa: E402
from resid_knn import CONFIGS, KSHR, spaces  # noqa: E402
from select_corr import CACHE, EPS, GRID, SPEC, cv_rmse  # noqa: E402

ALT = {"gas": ("rel", "pl"), "eui": ("rel", "pl"), "elec": ("pl", "ar")}
FINAL = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[1] == "final" else None


def knn_multi(Zf, sf, Za, sa, NUM, DEN, W, k, spatial=0.0):
    """resid_knn.knn_mean for several (num, den) column pairs at once (same neighbour sets); returns (n_app, q)"""
    o = np.argsort(sf, kind="stable")
    so = sf[o]
    lo, hi = np.searchsorted(so, sa - W, "left"), np.searchsorted(so, sa + W, "right")
    L = int((hi - lo).max())
    chunk = max(256, int(2e7 // (L * (Zf.shape[1] + 1 + NUM.shape[1]))))
    numo, deno, Zo, sof = NUM[o], DEN[o], Zf[o], so.astype(float)
    out, cnt = np.zeros((len(sa), NUM.shape[1])), np.zeros(len(sa))
    for s in range(0, len(sa), chunk):
        idx = np.arange(s, min(s + chunk, len(sa)))
        valid = np.arange(L)[None, :] < (hi[idx] - lo[idx])[:, None]
        cand = np.where(valid, lo[idx, None] + np.arange(L)[None, :], 0)
        dist = ((Zo[cand] - Za[idx][:, None, :]) ** 2).sum(-1)
        if spatial:
            dist = dist + spatial * np.abs(sof[cand] - sa[idx, None]) / max(W, 1)
        dist = np.where(valid, dist, np.inf)
        kk = min(k, L)
        part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        sel, ok = np.take_along_axis(cand, part, 1), np.isfinite(np.take_along_axis(dist, part, 1))
        sn, sd = (numo[sel] * ok[..., None]).sum(1), (deno[sel] * ok[..., None]).sum(1)
        out[idx] = np.where(sd > 0, sn / np.where(sd > 0, sd, 1), 0.0)
        cnt[idx] = ok.sum(1)
    return (cnt / (cnt + KSHR))[:, None] * out


def estimator(est, metric, r, pP, pH, AP, AH):
    """(num, den, mult) of an estimator; 'base' = the metric's current estimator"""
    if est == "base":
        est = {"gas": "ar", "eui": "aw", "elec": "rel"}[metric]
    return {"ar": (r, AP, AH), "aw": (AP * r, AP, np.ones(len(AH))), "rel": (r, pP, pH),
            "pl": (r, np.ones(len(AP)), np.ones(len(AH)))}[est]


def chain():
    from train_lib import Data
    d = Data()
    tr, te = load()
    P, H = holdout_split(len(tr))
    kt, _ = group_keys(tr, te, d.subn, d.subnt)
    kP, kH = tuple(k[P] for k in kt), tuple(k[H] for k in kt)
    is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
    mem = sorted(f[:-4] for f in os.listdir("exp/hold/pred") if os.path.getsize(f"exp/hold/pred/{f}") > 0)
    gas = [X8.get(m, m) for m in mem if not is_el(m)] + ["p1_gpm"]
    el = [m for m in mem if is_el(m)] + ["p1_lepm"]
    path = lambda n, part: next(f for f in (f"exp/hold/{part}/{n}.npy", f"exp/hold/tune/{part}/{n}.npy")  # noqa: E731
                                if os.path.exists(f))
    A = tr["bld_dim_06"].values
    M = tr["bld_dim_12"].values.astype(float)
    res = {}
    for metric, col in METRICS.items():
        names = el if metric == "elec" else gas
        Po = np.column_stack([np.load(path(n, "oof")) for n in names])
        Ph = np.column_stack([np.load(path(n, "pred")) for n in names])
        yv = tr[EL if metric == "elec" else EUI].values
        sc = A[P] if metric == "gas" else None
        pP, pH, _ = blend_vc(Po, Ph, yv[P], SPEC, dict(A=A[P], M=M[P]), dict(A=A[H], M=M[H]), sc)
        if metric == "gas":
            pP, pH = pP * A[P], pH * A[H]
        scale = A if metric == "gas" else np.ones(len(A))
        rP, rH = scale[P] * yv[P] - pP, scale[H] * yv[H] - pH
        G = features(metric, kP, kH, rP, pP, pH, A[P], A[H])
        res[metric] = dict(rP=rP, rH=rH, pP=pP, pH=pH, G=G, sig=SIGMAS[col])
    return tr, te, d, P, H, A, res


def columns(tr, te, d, P, H, A, res):
    """alternative-estimator columns for every grid config (cached), plus a check of the base estimator vs cache"""
    Zs = spaces(tr, te)
    for cfg in GRID:
        files = {(m, e): f"{CACHE}/est_{m}_{e}_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}.npy" for m in ALT for e in ALT[m]}
        if all(os.path.exists(f) for f in files.values()):
            continue
        pairs = [(m, e) for m in ALT for e in ("base",) + ALT[m]]
        nd = [estimator(e, m, res[m]["rP"], res[m]["pP"], res[m]["pH"], A[P], A[H]) for m, e in pairs]
        sp, W, k, s = cfg
        out = knn_multi(Zs[sp][P], d.subn[P], Zs[sp][H], d.subn[H], np.column_stack([x[0] for x in nd]),
                        np.column_stack([x[1] for x in nd]), W, k, s)
        for j, ((m, e), x) in enumerate(zip(pairs, nd)):
            v = out[:, j] * x[2]
            if e == "base":
                ref = np.load(f"{CACHE}/{m}_{sp}_{W}_{k}_{s}.npy")
                assert np.allclose(v, ref, rtol=1e-9, atol=1e-9 * np.abs(ref).max()), (m, cfg)
            else:
                np.save(files[(m, e)], v)
        print("columns", cfg, flush=True)


def current(metric):
    if metric == "eui":
        return [tuple(c) for c in json.load(open("exp/analysis/corr_final.json"))["eui"]["configs"]]
    return [tuple(c) for c in CONFIGS[metric]]


def greedy(F0, cols, r, sig):
    sel, best = [], cv_rmse(F0, r, sig)
    while True:
        trial = sorted((cv_rmse(np.column_stack([F0] + [cols[c] for c in sel + [cand]]), r, sig), cand)
                       for cand in cols if cand not in sel)
        if not trial or trial[0][0] > best - EPS:
            return sel
        best = trial[0][0]
        sel.append(trial[0][1])


def main():
    tr, te, d, P, H, A, res = chain()
    columns(tr, te, d, P, H, A, res)
    halves = np.random.default_rng(11).permutation(len(H)) % 2
    final = {}
    for metric in METRICS:
        R = res[metric]
        cur = current(metric)
        F0 = np.column_stack([R["G"]] + [np.load(f"{CACHE}/{metric}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in cur])
        cols = {(e,) + c: np.load(f"{CACHE}/est_{metric}_{e}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy")
                for e in ALT[metric] for c in GRID}
        if FINAL is not None:
            if metric in FINAL:
                sel = greedy(F0, cols, R["rH"], R["sig"])
                print(f"{metric} final: {cv_rmse(F0, R['rH'], R['sig']):.6f} -> "
                      f"{cv_rmse(np.column_stack([F0] + [cols[c] for c in sel]), R['rH'], R['sig']):.6f} +{sel}",
                      flush=True)
                final[metric] = {"configs": [list(c) for c in cur] + [list(c[1:]) + [c[0]] for c in sel]}
            else:
                final[metric] = {"configs": [list(c) for c in cur]}
            continue
        out = []
        for h in (0, 1):
            a, b = halves == h, halves != h
            sel = greedy(F0[a], {c: v[a] for c, v in cols.items()}, R["rH"][a], R["sig"])
            new = cv_rmse(np.column_stack([F0[b]] + [cols[c][b] for c in sel]), R["rH"][b], R["sig"])
            old = cv_rmse(F0[b], R["rH"][b], R["sig"])
            out.append(new - old)
            print(f"{metric} half {h}: +{len(sel)} alt columns {sel} -> other half {new:.5f} vs {old:.5f} "
                  f"({new - old:+.6f})", flush=True)
        print(f"{metric} mean honest change {np.mean(out):+.6f}", flush=True)
    if FINAL is not None:
        json.dump(final, open("exp/analysis/corr_est_final.json", "w"), indent=1)
        print("wrote exp/analysis/corr_est_final.json", flush=True)


if __name__ == "__main__":
    main()
