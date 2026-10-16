"""Group + similarity-kNN residual correction of the blend (extends resid_corr.py).

On top of the zone / climate cell / sub_n window group means of resid_corr.py, each row gets the shrunken mean
OOF residual of the k fit rows most similar to it (in a building-stock feature space) within a +-W window of sub_n.
Rows of similar housing stock in the same area share effects the models miss. Neighbours chosen by sub_n distance
only add nothing; the gain comes from the similarity.

The weights are fitted on the holdout exactly as in resid_corr.py: H rows play the test rows, P-OOF residuals play
the train OOF residuals. Cross-fitted H gain over the group correction alone (2026-10-15 sweep): gas -0.0008,
EUI -0.0012, elec -0.0003.

usage: python src/resid_knn.py fit [--lam f]             holdout fit -> f (+ H gains)
       python src/resid_knn.py apply <tag> <new> [--lam f]   blend lists of exp/prune_<tag>.log -> sub_<new>.csv
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SIGMAS, TARGET_COLS, holdout_split, load  # noqa: E402
from prune import ELEC  # noqa: E402
from resid_corr import EL, EUI, METRICS, X8, blend_preds, features, group_keys  # noqa: E402

KSHR = 5.0  # shrinkage n / (n + KSHR) of a kNN mean
# per metric: (similarity space, window W, k, spatial distance weight), from the greedy holdout selection
# (scratchpad knn_resid3.py: 17 candidates, forward selection on cross-fitted H gain)
CONFIGS = {
    "gas": [("full", 100, 20, 4.0), ("era", 30, 20, 0.0), ("fc", 300, 50, 0.0), ("full", 10, 10, 0.0),
            ("era", 300, 50, 0.0), ("full", 3000, 200, 0.0)],
    "eui": [("full", 30, 10, 0.0), ("fc", 300, 50, 0.0), ("full", 3000, 200, 0.0), ("era", 30, 20, 0.0),
            ("full", 10, 10, 0.0), ("small", 30, 20, 0.0)],
    "elec": [("full", 100, 20, 4.0), ("full", 1000, 100, 0.0), ("full", 30, 20, 0.0), ("era", 300, 50, 0.0),
             ("small", 30, 20, 0.0)],
}


def spaces(tr, te):
    """similarity spaces for train + test rows, standardized with train statistics"""
    both = pd.concat([tr, te])
    n = len(tr)
    A, M, E = both["bld_dim_06"].values, both["bld_dim_12"].values.astype(float), both["bld_dim_01"].values.astype(float)
    era = both[[f"bld_era_{i:02d}" for i in range(9, 16)]].values / 100

    def std(Z):
        return (Z - Z[:n].mean(0)) / Z[:n].std(0)

    full = std(np.column_stack([np.log(A / M), np.log(E / M), np.log(M), era, both["bld_dim_02"].values / E,
                                np.log1p(both["bld_dim_17"].values), np.log(both["bld_dim_03"].values / A)]))
    soc_cols = [c for c in tr.columns if c.startswith(("socio_demog", "hvac_dist"))
                and pd.api.types.is_numeric_dtype(tr[c]) and tr[c].nunique() > 1]
    soc = std(both[soc_cols].values)
    pca = PCA(8, random_state=0).fit(soc[:n])
    cen = std(pca.transform(soc))
    return {"full": full, "era": std(np.column_stack([np.log(A / M), era])),
            "small": std(np.column_stack([np.log(A / M), np.log(E / M), np.log(M)])),
            "cen": cen, "fc": np.column_stack([full, cen * 0.5])}


def knn_mean(Zf, sf, Za, sa, num, den, W, k, spatial=0.0):
    """sum(num)/sum(den) over the k fit rows nearest to each apply row in Z (+ spatial * |sub_n diff| / W) among
    fit rows with |sub_n diff| <= W, shrunk by n/(n+KSHR)"""
    o = np.argsort(sf, kind="stable")
    so = sf[o]
    lo, hi = np.searchsorted(so, sa - W, "left"), np.searchsorted(so, sa + W, "right")
    L = int((hi - lo).max())
    chunk = max(256, int(2e7 // (L * (Zf.shape[1] + 1))))
    numo, deno, Zo, sof = num[o], den[o], Zf[o], so.astype(float)
    out, cnt = np.zeros(len(sa)), np.zeros(len(sa))
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
        sn, sd = (numo[sel] * ok).sum(1), (deno[sel] * ok).sum(1)
        out[idx] = np.where(sd > 0, sn / np.where(sd > 0, sd, 1), 0.0)
        cnt[idx] = ok.sum(1)
    return cnt / (cnt + KSHR) * out


def all_features(metric, kf, ka, r, p_fit, p_app, A_fit, A_app, Zs, f_rows, a_rows, sub_f, sub_a):
    """group features of resid_corr.py followed by the metric's kNN features (same metric scaling)"""
    F = features(metric, kf, ka, r, p_fit, p_app, A_fit, A_app)
    if metric == "gas":
        num, den, mult = r, A_fit, A_app
    elif metric == "eui":
        num, den, mult = A_fit * r, A_fit, 1.0
    else:
        num, den, mult = r, p_fit, p_app
    cols = [knn_mean(Zs[space][f_rows], sub_f, Zs[space][a_rows], sub_a, num, den, W, k, sp) * mult
            for space, W, k, sp in CONFIGS[metric]]
    return np.column_stack([F] + cols)


def fit(lam_file):
    from train_lib import Data
    d = Data()
    tr, te = load()
    P, H = holdout_split(len(tr))
    kt, _ = group_keys(tr, te, d.subn, d.subnt)
    kP, kH = tuple(k[P] for k in kt), tuple(k[H] for k in kt)
    Zs = spaces(tr, te)
    is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
    mem = sorted(f[:-4] for f in os.listdir("exp/hold/pred") if os.path.getsize(f"exp/hold/pred/{f}") > 0)
    gas = [X8.get(m, m) for m in mem if not is_el(m)] + ["p1_gpm"]
    el = [m for m in mem if is_el(m)] + ["p1_lepm"]
    path = lambda n, part: next(f for f in (f"exp/hold/{part}/{n}.npy", f"exp/hold/tune/{part}/{n}.npy")  # noqa: E731
                                if os.path.exists(f))
    A = tr["bld_dim_06"].values
    lam = {}
    for metric, col in METRICS.items():
        names = el if metric == "elec" else gas
        Po = np.column_stack([np.load(path(n, "oof")) for n in names])
        Ph = np.column_stack([np.load(path(n, "pred")) for n in names])
        yv = tr[EL if metric == "elec" else EUI].values
        pP, pH = blend_preds(Po, Ph, yv[P], metric, A[P], A[H])
        scale = A if metric == "gas" else np.ones(len(A))
        rP, rH = scale[P] * yv[P] - pP, scale[H] * yv[H] - pH
        F = all_features(metric, kP, kH, rP, pP, pH, A[P], A[H], Zs, P, H, d.subn[P], d.subn[H])
        base = np.sqrt(np.mean(rH ** 2)) / SIGMAS[col]
        msg = []
        for nf in (3, F.shape[1]):
            pc = np.zeros(len(H))
            for a, b in KFold(5, shuffle=True, random_state=0).split(F):
                pc[b] = F[b, :nf] @ np.linalg.lstsq(F[a, :nf], rH[a], rcond=None)[0]
            msg.append(f"{np.sqrt(np.mean((rH - pc) ** 2)) / SIGMAS[col]:.5f}")
        lam[metric] = np.linalg.lstsq(F, rH, rcond=None)[0].tolist()
        print(f"{metric:4s} H {base:.5f} -> group {msg[0]} -> group+knn {msg[1]}  lam {np.round(lam[metric], 3)}",
              flush=True)
    json.dump({"lam": lam, "configs": CONFIGS, "kshr": KSHR}, open(lam_file, "w"), indent=1)
    print("wrote", lam_file)


def apply(tag, new, lam_file):
    from train_lib import Data
    d = Data()
    tr, te = load()
    kt, ke = group_keys(tr, te, d.subn, d.subnt)
    cfg = json.load(open(lam_file))
    assert cfg["configs"] == {m: [list(c) for c in v] for m, v in CONFIGS.items()} and cfg["kshr"] == KSHR
    Zs = spaces(tr, te)
    n = len(tr)
    f_rows, a_rows = np.arange(n), np.arange(n, n + len(te))
    log = open(f"exp/prune_{tag}.log").read()
    runs = {k: re.search(rf"^{k} runs: (\S+)", log, re.M).group(1).split(",") for k in ("gas", "elec")}
    sub = pd.read_csv(f"submissions/sub_{tag}.csv")
    A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
    for metric, col in METRICS.items():
        names = runs["elec" if metric == "elec" else "gas"]
        Po = np.column_stack([np.load(f"exp/oof/{nm}.npy") for nm in names])
        Pt = np.column_stack([np.load(f"exp/pred/{nm}.npy") for nm in names])
        yv = tr[EL if metric == "elec" else EUI].values
        pf, pa = blend_preds(Po, Pt, yv, metric, A, At)
        assert np.allclose(np.clip(pa, 0, None), sub[col].values, rtol=1e-6, atol=1e-6), f"{col}: blend differs"
        scale = A if metric == "gas" else np.ones(len(A))
        r = scale * yv - pf
        F = all_features(metric, kt, ke, r, pf, pa, A, At, Zs, f_rows, a_rows, d.subn, d.subnt)
        corr = F @ np.array(cfg["lam"][metric])
        sub[col] = np.clip(pa + corr, 0, None)
        print(f"{metric:4s} correction rms/sigma {np.sqrt(np.mean(corr ** 2)) / SIGMAS[col]:.5f}", flush=True)
    assert np.isfinite(sub[TARGET_COLS].values).all()
    out = f"submissions/sub_{new}.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "apply"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--lam", default="exp/resid_knn.json")
    a = ap.parse_args()
    if a.mode == "fit":
        fit(a.lam)
    else:
        apply(*a.tags, a.lam)
