"""Post-processing of the blend: group + similarity-kNN residual correction (frozen weights), copied verbatim from
src/resid_corr.py and src/resid_knn.py (development versions, which also contain the holdout fit of the weights).

The correction of a test row is a linear combination (weights lambda in configs/resid_knn.json, fitted on an internal
70/30 holdout of train) of shrunken means of the blend's OOF residuals of train rows that share its zone, climate cell
or +-30 sub_n window, and of the k most similar train rows (building-stock feature space) within a +-W sub_n window.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from blend import nnls_blend

K = 30    # shrinkage n / (n + K) of a group mean
WIN = 30  # half-width of the sub_n window
KSHR = 5.0  # shrinkage n / (n + KSHR) of a kNN mean
CONFIGS = {
    "gas": [("full", 100, 20, 4.0), ("era", 30, 20, 0.0), ("fc", 300, 50, 0.0), ("full", 10, 10, 0.0),
            ("era", 300, 50, 0.0), ("full", 3000, 200, 0.0)],
    "eui": [("full", 30, 10, 0.0), ("fc", 300, 50, 0.0), ("full", 3000, 200, 0.0), ("era", 30, 20, 0.0),
            ("full", 10, 10, 0.0), ("small", 30, 20, 0.0)],
    "elec": [("full", 100, 20, 4.0), ("full", 1000, 100, 0.0), ("full", 30, 20, 0.0), ("era", 300, 50, 0.0),
             ("small", 30, 20, 0.0)],
}


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
