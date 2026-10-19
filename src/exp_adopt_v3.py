"""PP0005: adoption v3 A/B for the unseen-subzone model (count_r = households in the missing heating category).

Same 5-fold subzone CV and unseen-weighted RMSE as src/adopt_v2.py (test row mix of the unseen subzones). Variants:
- base   : adopt_v2 features (q_xent + r_reg average), reproduces 1.0303;
- cknn   : + pooled count_r / K of the k most similar labelled subzones in a census space (PCA of the standardized
           subzone-level census columns), out-of-fold like the zone / window pools;
- wmix   : training weights that move the nrow mix of the labelled subzones to the unseen test subzone mix;
- wrow   : as wmix but matching the unseen test row mix (the weighting of the score);
- epc    : + subzone sums of the postcode-level building columns (train + test rows) and their ratios to N and K;
Out-of-fold r predictions are saved in exp/adopt/ so that the variants can be blended on the same folds.

usage: python src/exp_adopt_v3.py <variant>[,<variant>...] [--smoke]   (--smoke: 20k subzones, quick logic check)
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(__file__))
from adopt_v2 import BASE, UNSEEN_MIX, WINDOWS, cv_scores, neighbour_feats, subzone_table  # noqa: E402
from common import HVAC, SEED, load  # noqa: E402

OUT = "exp/adopt"
KS = (10, 30, 100)
NB_MAX = 400  # neighbours kept per subzone before the fold filter
THREADS = 4


def census_neighbours(S, lab):
    """indices (into the labelled subzones) of the NB_MAX nearest labelled subzones of every subzone, in a PCA space
    of the standardized subzone-level census columns (no target involved)"""
    cols = sorted([c for c in S.columns if c.startswith(("socio_demog", "hvac_dist")) and not c.endswith("_m")])
    Z = S[cols].values.astype(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    Z = PCA(30, random_state=SEED).fit_transform(Z).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=NB_MAX + 1, algorithm="brute", n_jobs=THREADS).fit(Z[lab])
    _, idx = nn.kneighbors(Z)
    return idx  # (n_subzones, NB_MAX + 1), positions in S[lab]


def cknn_feats(nb, q_rows, self_pos, allowed, K, cr):
    """pooled count_r / K over the first k allowed neighbours (k in KS) for the query rows q_rows (positions in S).
    nb: neighbour positions in the labelled table; self_pos: own labelled position or -1; allowed: bool over labelled"""
    cand = nb[q_rows]
    ok = allowed[cand] & (cand != self_pos[:, None])
    rank = np.cumsum(ok, axis=1)
    out = {}
    for k in KS:
        take = ok & (rank <= k)
        sk = np.where(take, K[cand], 0).sum(1)
        sc = np.where(take, cr[cand], 0).sum(1)
        out[f"ck{k}_q"] = np.where(sk > 0, sc / np.maximum(sk, 1), np.nan)
        out[f"ck{k}_mq"] = np.where(take, cr[cand] / np.maximum(K[cand], 1), 0).sum(1) / np.maximum(take.sum(1), 1)
    return pd.DataFrame(out)


def mix_weights(L, rows=False):
    """per-subzone weight that turns the labelled nrow mix into the unseen test subzone mix (rows=True: row mix)"""
    b = np.minimum(L.nrow.values, 6)
    tgt = pd.Series({k: (v if rows else v / k) for k, v in UNSEEN_MIX.items()})
    tgt = tgt / tgt.sum()
    have = pd.Series(b).value_counts(normalize=True)
    w = (tgt / have).reindex(range(1, 7)).fillna(0)
    return w.loc[b].values


def fit_one(Fa, La, Fb, Lb, Fu, U, mode, wa=None, wb=None, params=None):
    p = dict(BASE, num_threads=THREADS, **(params or {}))
    wa = np.ones(len(La)) if wa is None else wa
    wb = np.ones(len(Lb)) if wb is None else wb
    if mode == "q_xent":
        ma, mb = La.K.values > 0, Lb.K.values > 0
        qa, qb = (La.cr / La.K.clip(lower=1)).values, (Lb.cr / Lb.K.clip(lower=1)).values
        m = lgb.train(dict(p, objective="cross_entropy"), lgb.Dataset(Fa[ma], qa[ma], weight=(La.K.values * wa)[ma]),
                      5000, valid_sets=[lgb.Dataset(Fb[mb], qb[mb], weight=(Lb.K.values * wb)[mb])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])

        def to_r(F, L):
            return 100 * L.K.values * np.clip(m.predict(F, num_iteration=m.best_iteration), 0, 1) / L.N.values
    else:
        m = lgb.train(dict(p, objective="regression"), lgb.Dataset(Fa, (La.rest - La.ad).values, weight=wa), 5000,
                      valid_sets=[lgb.Dataset(Fb, (Lb.rest - Lb.ad).values, weight=wb)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])

        def to_r(F, L):
            return np.clip(m.predict(F, num_iteration=m.best_iteration), 0, None)
    return to_r(Fb, Lb), to_r(Fu, U), m.best_iteration


def main(variants, smoke=False):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tr, te = load()
    S = subzone_table(tr, te)
    if any("epc" in v for v in variants):
        al = pd.concat([tr, te], ignore_index=True)
        num = [c for c in al.columns if c.startswith(("bld_dim", "bld_era")) and al[c].dtype.kind in "if"]
        pc = sorted(c for c in num if (al.groupby("subzone_id")[c].nunique() > 1).any())
        sums = al.groupby("subzone_id")[pc].sum().reindex(S.index)
        epc = sums.add_suffix("_s")
        for c in pc:
            epc[c + "_sN"] = sums[c] / S.N.clip(lower=1)
            epc[c + "_sK"] = sums[c] / S.K.clip(lower=1)
        S = S.join(epc)
        print(f"epc features {epc.shape[1]}", flush=True)
    if smoke:
        S = S.sample(20000, random_state=SEED)
        variants = [v + "_smoke" for v in variants]
    lab = S.ad.notna().values
    L, U = S[lab], S[~lab]
    size = int(S.sub_n.max()) + 2 * max(WINDOWS) + 2
    base_feats = [c for c in S.columns if c not in ("zone", "cell", "ad", "cr")]
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(L))
    need_knn = any("cknn" in v for v in variants)
    if need_knn:
        nb = census_neighbours(S, lab)
        pos_l = np.full(len(S), -1)
        pos_l[np.where(lab)[0]] = np.arange(lab.sum())
        nb_l, nb_u = nb[lab], nb[~lab]
        Kl, crl = L.K.values.astype(float), L.cr.values.astype(float)
        print(f"census neighbours done {time.time() - t0:.0f}s", flush=True)
    wl, wr = mix_weights(L), mix_weights(L, rows=True)
    for v in variants:
        feats = [c for c in base_feats if "epc" in v or not c.endswith(("_s", "_sN", "_sK"))]
        modes = ("q_xent", "r_reg")
        oof = {m: np.zeros(len(L)) for m in modes}
        pu = {m: np.zeros(len(U)) for m in modes}
        for k, (a, b) in enumerate(folds):
            La, Lb = L.iloc[a], L.iloc[b]
            inner = list(KFold(5, shuffle=True, random_state=SEED).split(La))
            na = pd.concat([neighbour_feats(La.iloc[ia], La.iloc[ib], size) for ia, ib in inner]).reindex(La.index)
            Fa = La[feats].join(na)
            Fb = Lb[feats].join(neighbour_feats(La, Lb, size))
            Fu = U[feats].join(neighbour_feats(La, U, size))
            if "cknn" in v:
                in_a = np.zeros(len(L), bool)
                in_a[a] = True
                parts = []
                for ia, ib in inner:
                    allowed = np.zeros(len(L), bool)
                    allowed[a[ia]] = True
                    f = cknn_feats(nb_l, a[ib], a[ib], allowed, Kl, crl)
                    f.index = La.index[ib]
                    parts.append(f)
                Fa = Fa.join(pd.concat(parts).reindex(La.index))
                fb = cknn_feats(nb_l, b, b, in_a, Kl, crl)
                fb.index = Lb.index
                Fb = Fb.join(fb)
                fu = cknn_feats(nb_u, np.arange(len(U)), np.full(len(U), -1), in_a, Kl, crl)
                fu.index = U.index
                Fu = Fu.join(fu)
            w = wl if "wmix" in v else (wr if "wrow" in v else None)
            wa = None if w is None else w[a]
            wb = None if w is None else w[b]
            for mode in modes:
                rb, ru, it = fit_one(Fa, La, Fb, Lb, Fu, U, mode, wa, wb)
                oof[mode][b] = rb
                pu[mode] += ru / len(folds)
                print(f"  [{v}] fold {k} {mode} it={it} {time.time() - t0:.0f}s", flush=True)
        for m in modes:
            np.save(f"{OUT}/{v}_{m}_oof.npy", oof[m])
            np.save(f"{OUT}/{v}_{m}_unseen.npy", pu[m])
            s = cv_scores(L, oof[m])
            print(f"[{v}] {m}: CV rmse all {s[0]:.4f} unseen-weighted {s[1]:.4f}", flush=True)
        s = cv_scores(L, np.mean([oof[m] for m in modes], axis=0))
        print(f"[{v}] average: CV rmse all {s[0]:.4f} unseen-weighted {s[1]:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","), smoke="--smoke" in sys.argv)
