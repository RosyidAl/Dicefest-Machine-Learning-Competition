"""kNN neighbour statistics within sub_n windows (FB07), shared by train_lib and src/features."""
import numpy as np


def knn_stats(Zf, sf, Vf, Za, sa, W, k, spatial=0.0, exclude_self=None):
    """mean of the value columns Vf over the k fit rows nearest to each apply row (squared distance in Z plus
    spatial * |sub_n diff| / W) among fit rows with |sub_n diff| <= W; also the neighbour count and mean distance.
    exclude_self: for each apply row, the fit-row index to skip (or -1)."""
    o = np.argsort(sf, kind="stable")
    so = sf[o]
    lo, hi = np.searchsorted(so, sa - W, "left"), np.searchsorted(so, sa + W, "right")
    L = max(int((hi - lo).max()), 1)
    chunk = max(256, int(2e7 // (L * (Zf.shape[1] + 1))))
    Vo, Zo, sof = Vf[o], Zf[o], so.astype(float)
    inv = np.empty(len(o), dtype=np.int64)
    inv[o] = np.arange(len(o))
    out = np.full((len(sa), Vf.shape[1]), np.nan)
    cnt, md = np.zeros(len(sa)), np.full(len(sa), np.nan)
    for s in range(0, len(sa), chunk):
        idx = np.arange(s, min(s + chunk, len(sa)))
        valid = np.arange(L)[None, :] < (hi[idx] - lo[idx])[:, None]
        cand = np.where(valid, lo[idx, None] + np.arange(L)[None, :], 0)
        if exclude_self is not None:
            es = exclude_self[idx]
            valid &= ~((es[:, None] >= 0) & (cand == np.where(es >= 0, inv[np.maximum(es, 0)], -1)[:, None]))
        dist = ((Zo[cand] - Za[idx][:, None, :]) ** 2).sum(-1)
        if spatial:
            dist = dist + spatial * np.abs(sof[cand] - sa[idx, None]) / max(W, 1)
        dist = np.where(valid, dist, np.inf)
        kk = min(k, L)
        part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        sel = np.take_along_axis(cand, part, 1)
        dsel = np.take_along_axis(dist, part, 1)
        ok = np.isfinite(dsel)
        c = ok.sum(1)
        good = c > 0
        out[idx[good]] = (Vo[sel] * ok[..., None]).sum(1)[good] / c[good, None]
        cnt[idx] = c
        md[idx[good]] = np.where(ok, dsel, 0).sum(1)[good] / c[good]
    return out, cnt, md
