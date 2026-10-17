"""FB07b: extra kNN-target statistics appended to the FB07 block of train_lib.Data (A/B: src/exp_fb07b.py).

The extra neighbour spaces / windows are the ones the residual correction found useful (src/resid_knn.py CONFIGS):
building era space, building + census PCA space, a 3-feature size space, a very local (W=10) and a wide (W=1000)
window in the full space. Set `d.sp = {name: (Z_train_rows, Z_apply_rows)}` (src/resid_knn.spaces) before use.

The FB07 part comes from Data.knn_features (cached in d.knn_cache_dir, shared with the v16 members); the extra part is
cached separately in d.knn_x_cache_dir. Apply rows are split into blocks that run in parallel threads (numpy releases
the GIL); every row is computed independently, so the values do not depend on the number of threads.
"""
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED  # noqa: E402
from features_knn import knn_stats  # noqa: E402
from train_lib import Data  # noqa: E402

EXTRA = (("era", 30, 20, 0.0), ("era", 300, 50, 0.0), ("fc", 300, 50, 0.0), ("full", 10, 10, 0.0),
         ("full", 1000, 100, 0.0), ("small", 30, 20, 0.0))
THREADS = 8
BLOCK = 20000  # apply rows per parallel task


def _blocked(configs, Zf, sf, Vf, Za_of, sa_of, n_apply, space_of):
    """knn_stats for every config over blocks of apply rows in parallel threads; returns one result per config"""
    blocks = [np.arange(s, min(s + BLOCK, n_apply)) for s in range(0, n_apply, BLOCK)]

    def one(task):
        cfg, blk = task
        sp_name = space_of(cfg)
        return knn_stats(Zf(sp_name), sf, Vf, Za_of(sp_name, blk), sa_of(blk), cfg[-3], cfg[-2], cfg[-1])

    tasks = [(cfg, blk) for cfg in configs for blk in blocks]
    with ThreadPoolExecutor(THREADS) as ex:
        res = list(ex.map(one, tasks))
    out = []
    for i in range(len(configs)):
        parts = res[i * len(blocks):(i + 1) * len(blocks)]
        out.append(tuple(np.concatenate([p[j] for p in parts]) for j in range(3)))
    return out


class DataP(Data):
    """Data with the FB07 block computed in parallel blocks (same configs and values as Data._knn_features up to
    distance ties); used by the A/B so that both variants share the same, faster FB07 code"""

    def _knn_features(self, fit_idx, apply="oof", inner=5):
        Z, Zt = self.knn_space()
        V = np.c_[np.log(self.V[:, 0]), self.V[:, 3], np.log(self.V[:, 5])]

        def feats(f, rows_apply, test):
            n_apply = len(self.subnt) if test else len(rows_apply)
            res = _blocked([("full",) + c for c in self.KNN_CONFIGS], lambda _: Z[f], self.subn[f], V[f],
                           lambda _, blk: Zt[blk] if test else Z[rows_apply[blk]],
                           lambda blk: self.subnt[blk] if test else self.subn[rows_apply[blk]], n_apply, lambda c: c[0])
            cols = {}
            for (W, k, sp), (v, c, dd) in zip(self.KNN_CONFIGS, res):
                tag = f"knn_w{W}_k{k}" + ("s" if sp else "")
                for j, nm in enumerate(("leui", "lepm", "lgpm")):
                    cols[f"{tag}_{nm}"] = v[:, j]
                cols[f"{tag}_cnt"] = c
                cols[f"{tag}_dist"] = dd
            return pd.DataFrame(cols).astype("float32")

        return _cross_fit(feats, fit_idx, apply, inner)


def _cross_fit(feats, fit_idx, apply, inner):
    """inner-OOF features for fit rows ('oof'), features of test rows ('test') or of train rows (('train', idx))"""
    if isinstance(apply, str) and apply == "oof":
        parts, idx = [], []
        for ia, ib in KFold(inner, shuffle=True, random_state=SEED).split(fit_idx):
            parts.append(feats(fit_idx[ia], fit_idx[ib], False))
            idx.append(ib)
        F = pd.concat(parts, ignore_index=True)
        return F.iloc[np.argsort(np.concatenate(idx), kind="stable")].reset_index(drop=True)
    if isinstance(apply, str) and apply == "test":
        return feats(fit_idx, None, True)
    return feats(fit_idx, apply[1], False)


class DataX(Data):
    """Data whose kNN block is FB07 followed by FB07b (train rows = self.X, apply rows = self.Xt)"""

    def knn_features(self, fit_idx, apply="oof", inner=5):
        F = super().knn_features(fit_idx, apply, inner)
        cdir = getattr(self, "knn_x_cache_dir", None)
        path = None
        if cdir:
            h = hashlib.sha256()
            for arr in (np.asarray(fit_idx), self.subn, self.subnt, self.V[:, [0, 3, 5]]):
                h.update(np.ascontiguousarray(arr).tobytes())
            for name in sorted(self.sp):
                h.update(np.ascontiguousarray(self.sp[name][0][:1000]).tobytes())
            h.update(repr((apply if isinstance(apply, str) else ("train",), inner, EXTRA)).encode())
            if not isinstance(apply, str):
                h.update(np.ascontiguousarray(apply[1]).tobytes())
            path = os.path.join(cdir, h.hexdigest()[:32] + ".parquet")
            if os.path.exists(path):
                return pd.concat([F.reset_index(drop=True), pd.read_parquet(path)], axis=1)
        Fx = self._knn_extra(fit_idx, apply, inner)
        if path:
            os.makedirs(cdir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            Fx.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        return pd.concat([F.reset_index(drop=True), Fx], axis=1)

    def _knn_extra(self, fit_idx, apply="oof", inner=5):
        V = np.c_[np.log(self.V[:, 0]), self.V[:, 3], np.log(self.V[:, 5])]  # leui, lepm, lgpm

        def feats(f, rows_apply, test):
            n_apply = len(self.subnt) if test else len(rows_apply)
            blocks = [np.arange(s, min(s + BLOCK, n_apply)) for s in range(0, n_apply, BLOCK)]

            def one(task):
                (name, W, k, sp), blk = task
                Ztr, Zte = self.sp[name]
                Za = Zte[blk] if test else Ztr[rows_apply[blk]]
                sa = self.subnt[blk] if test else self.subn[rows_apply[blk]]
                return knn_stats(Ztr[f], self.subn[f], V[f], Za, sa, W, k, sp)

            tasks = [(cfg, blk) for cfg in EXTRA for blk in blocks]
            with ThreadPoolExecutor(THREADS) as ex:
                res = list(ex.map(one, tasks))
            cols, i = {}, 0
            for name, W, k, sp in EXTRA:
                parts = res[i:i + len(blocks)]
                i += len(blocks)
                v = np.concatenate([p[0] for p in parts])
                dd = np.concatenate([p[2] for p in parts])
                tag = f"kx_{name}_w{W}_k{k}"
                for j, nm in enumerate(("leui", "lepm", "lgpm")):
                    cols[f"{tag}_{nm}"] = v[:, j]
                cols[f"{tag}_dist"] = dd
            return pd.DataFrame(cols).astype("float32")

        if isinstance(apply, str) and apply == "oof":
            parts, idx = [], []
            for ia, ib in KFold(inner, shuffle=True, random_state=SEED).split(fit_idx):
                parts.append(feats(fit_idx[ia], fit_idx[ib], False))
                idx.append(ib)
            F = pd.concat(parts, ignore_index=True)
            return F.iloc[np.argsort(np.concatenate(idx), kind="stable")].reset_index(drop=True)
        if isinstance(apply, str) and apply == "test":
            return feats(fit_idx, None, True)
        return feats(fit_idx, apply[1], False)
