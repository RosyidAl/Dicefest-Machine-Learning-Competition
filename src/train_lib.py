"""Cross-validated training harness: one call = one (model, target formulation) pair.

Every run writes out-of-fold predictions (oof/<name>.npy) and the fold-averaged test
prediction (pred/<name>.npy) on the *physical* scale of the modelled quantity, so runs
can be blended afterwards.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, TARGET_COLS, base_features, constant_cols, load, nrmse  # noqa: E402
from te import group_stats, oof_group_stats, oof_window_stats, stats_to_features, window_stats  # noqa: E402

N_FOLDS = 5


class Data:
    def __init__(self, cache="cache", n_folds=N_FOLDS):
        tr, te = load(cache)
        const = constant_cols(tr)
        self.X = base_features(tr, const)
        self.Xt = base_features(te, const)
        self.A, self.At = tr["bld_dim_06"].values, te["bld_dim_06"].values
        self.M, self.Mt = tr["bld_dim_12"].values, te["bld_dim_12"].values
        self.y = {c: tr[c].values for c in TARGET_COLS}
        self.sub, self.subt = tr["subzone_id"].values, te["subzone_id"].values
        self.zone, self.zonet = tr["zone_region_id"].values, te["zone_region_id"].values
        self.subn, self.subnt = self.X["sub_n"].values.astype(int), self.Xt["sub_n"].values.astype(int)
        self.subn_max = int(max(self.subn.max(), self.subnt.max()))
        self.ids = te["id"].values
        self.folds = list(KFold(n_folds, shuffle=True, random_state=SEED).split(self.X))
        y_gas, y_elec, y_eui = (self.y[c] for c in TARGET_COLS[:3])
        # quantities that get spatially encoded
        self.V = np.c_[y_eui, np.log(y_eui), y_elec / self.M, np.log(y_elec / self.M), y_elec / self.A,
                       y_gas / self.M, np.log(y_gas)]
        self.VN = ["eui", "leui", "epm", "lepm", "epa", "gpm", "lgas"]
        self.W = np.c_[np.log(self.A), np.log(self.M)]
        self.WN = ["lA", "lM"]
        # climate-grid cells: rows sharing all climate values lie in the same grid square (~15 rows each)
        cl = pd.concat([tr, te], ignore_index=True)
        n = len(tr)
        cells = [cl.groupby([f"env_climate_0{i}" for i in g], sort=False, dropna=False).ngroup().values
                 for g in ((1, 3, 5),)]  # climate 02/04 repeat this grid (with a 0.0 filler)
        self.cells, self.cellst = [c[:n] for c in cells], [c[n:] for c in cells]

    # ---- spatial encodings -------------------------------------------------
    def te_features(self, fit_idx, apply="oof", windows=(), cells=False, self_idx=None):
        """apply='oof' -> inner-OOF features for fit_idx; else a tuple of apply-row keys
        (subzone, zone, sub_n[, list of cell keys]).

        self_idx: when the apply rows are themselves part of fit_idx (leave-one-out use), their own
        values are removed from the subzone/zone statistics (windows and cells already skip the own subzone)."""
        vals = np.c_[self.V, self.W]
        f = []
        if isinstance(apply, str) and apply == "oof":
            s, c = oof_group_stats(self.sub[fit_idx], vals[fit_idx])
            sz, cz = oof_group_stats(self.zone[fit_idx], self.V[fit_idx])
            f += [stats_to_features(s, c, self.VN + self.WN, "sz"), stats_to_features(sz, cz, self.VN, "zn")]
            for w in windows:
                sw, cw = oof_window_stats(self.subn[fit_idx], self.V[fit_idx], w, self.subn_max)
                f.append(stats_to_features(sw, cw, self.VN, f"w{w}"))
            if cells:  # cell statistics excluding rows of the own subzone
                for j, cell in enumerate(self.cells):
                    key = cell[fit_idx].astype(np.int64)
                    s1, c1 = oof_group_stats(key, self.V[fit_idx])
                    s2, c2 = oof_group_stats(key * 10 ** 6 + self.subn[fit_idx], self.V[fit_idx])
                    f.append(stats_to_features(s1 - s2, c1 - c2, self.VN, f"c{j}x"))
        else:
            keys_sub, keys_zone, pos = apply[:3]
            s, c = group_stats(self.sub[fit_idx], vals[fit_idx], keys_sub)
            sz, cz = group_stats(self.zone[fit_idx], self.V[fit_idx], keys_zone)
            if self_idx is not None:
                s, c = s - vals[self_idx], c - 1
                sz, cz = sz - self.V[self_idx], cz - 1
            f += [stats_to_features(s, c, self.VN + self.WN, "sz"), stats_to_features(sz, cz, self.VN, "zn")]
            for w in windows:
                sw, cw = window_stats(self.subn[fit_idx], self.V[fit_idx], pos, w, self.subn_max)
                f.append(stats_to_features(sw, cw, self.VN, f"w{w}"))
            if cells:
                for j, (cell, ka) in enumerate(zip(self.cells, apply[3])):
                    key, ka = cell[fit_idx].astype(np.int64), ka.astype(np.int64)
                    s1, c1 = group_stats(key, self.V[fit_idx], ka)
                    s2, c2 = group_stats(key * 10 ** 6 + self.subn[fit_idx], self.V[fit_idx], ka * 10 ** 6 + pos)
                    f.append(stats_to_features(s1 - s2, c1 - c2, self.VN, f"c{j}x"))
        return pd.concat(f, axis=1).astype("float32")

    def te_apply(self, idx=None):
        """Apply-row keys for te_features: train rows idx, or the test set when idx is None."""
        if idx is None:
            return self.subt, self.zonet, self.subnt, self.cellst
        return self.sub[idx], self.zone[idx], self.subn[idx], [c[idx] for c in self.cells]

    # ---- FB07: kNN target encodings in the building-stock space ------------------------------------------
    KNN_CONFIGS = ((30, 20, 0.0), (100, 20, 4.0), (300, 50, 0.0))  # (sub_n window W, k, spatial distance weight)

    def knn_space(self):
        """standardised building-stock space of train rows (self.X) and apply rows (self.Xt); cached"""
        if getattr(self, "_kz", None) is None:
            def raw(X):
                A, M, E = X["bld_dim_06"].values, X["bld_dim_12"].values.astype(float), X["bld_dim_01"].values.astype(float)
                era = X[[f"bld_era_{i:02d}" for i in range(9, 16)]].values / 100
                return np.column_stack([np.log(A / M), np.log(E / M), np.log(M), era, X["bld_dim_02"].values / E,
                                        np.log1p(X["bld_dim_17"].values), np.log(X["bld_dim_03"].values / A)])
            Z, Zt = raw(self.X), raw(self.Xt)
            mu, sd = Z.mean(0), Z.std(0)
            self._kz = ((Z - mu) / sd, (Zt - mu) / sd)
        return self._kz

    def knn_features(self, fit_idx, apply="oof", inner=5):
        """apply='oof' -> inner 5-fold cross-fitted features for fit_idx rows; ('train', idx) -> train rows idx;
        'test' -> apply rows (self.Xt). Values: log EUI, log elec / meter, log gas / meter of fit rows.
        Results are cached in self.knn_cache_dir (if set), keyed by a hash of the inputs (identical either way)."""
        import hashlib
        cdir = getattr(self, "knn_cache_dir", None)
        if cdir:
            h = hashlib.sha256()
            for arr in (np.asarray(fit_idx), self.subn, self.subnt, self.V[:, [1, 3, 5]]):
                h.update(np.ascontiguousarray(arr).tobytes())
            h.update(repr((apply if isinstance(apply, str) else ("train",), inner, self.KNN_CONFIGS)).encode())
            if not isinstance(apply, str):
                h.update(np.ascontiguousarray(apply[1]).tobytes())
            path = os.path.join(cdir, h.hexdigest()[:32] + ".parquet")
            if os.path.exists(path):
                return pd.read_parquet(path)
            F = self._knn_features(fit_idx, apply, inner)
            os.makedirs(cdir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            F.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            return F
        return self._knn_features(fit_idx, apply, inner)

    def _knn_features(self, fit_idx, apply="oof", inner=5):
        from features_knn import knn_stats
        Z, Zt = self.knn_space()
        V = np.c_[np.log(self.V[:, 0]), self.V[:, 3], np.log(self.V[:, 5])]  # leui, lepm, lgpm

        def feats(f, Za, sa):
            cols = {}
            for W, k, sp in self.KNN_CONFIGS:
                v, c, dd = knn_stats(Z[f], self.subn[f], V[f], Za, sa, W, k, sp)
                tag = f"knn_w{W}_k{k}" + ("s" if sp else "")
                for j, nm in enumerate(("leui", "lepm", "lgpm")):
                    cols[f"{tag}_{nm}"] = v[:, j]
                cols[f"{tag}_cnt"] = c
                cols[f"{tag}_dist"] = dd
            return pd.DataFrame(cols).astype("float32")

        if isinstance(apply, str) and apply == "oof":
            parts, idx = [], []
            for ia, ib in KFold(inner, shuffle=True, random_state=SEED).split(fit_idx):
                parts.append(feats(fit_idx[ia], Z[fit_idx[ib]], self.subn[fit_idx[ib]]))
                idx.append(ib)
            F = pd.concat(parts, ignore_index=True)
            return F.iloc[np.argsort(np.concatenate(idx), kind="stable")].reset_index(drop=True)
        if isinstance(apply, str) and apply == "test":
            return feats(fit_idx, Zt, self.subnt)
        return feats(fit_idx, Z[apply[1]], self.subn[apply[1]])

    def agg_features(self, cache="cache"):
        """Subzone / zone aggregates over train and test rows: number of rows, log totals and row shares of
        meters, floor area and EPC count, mean log area per meter. Built from features only (no targets).
        Returns (train frame, test frame) for run(extra=..., extra_t=...)."""
        tr, te = load(cache)
        allr = pd.concat([tr, te], ignore_index=True)
        F = pd.DataFrame(index=allr.index)
        for gname in ("subzone_id", "zone_region_id"):
            g = allr.groupby(gname)
            t = gname[:3]
            F[f"{t}_cnt"] = g["id"].transform("size")
            for c, nm in (("bld_dim_12", "M"), ("bld_dim_06", "A"), ("bld_dim_01", "E")):
                s = g[c].transform("sum")
                F[f"{t}_sum{nm}"] = np.log1p(s)
                F[f"{t}_share{nm}"] = allr[c] / s
            F[f"{t}_lApM_mean"] = np.log(allr["bld_dim_06"] / allr["bld_dim_12"]).groupby(allr[gname]).transform("mean")
        F = F.astype("float32")
        n = len(tr)
        return F.iloc[:n].reset_index(drop=True), F.iloc[n:].reset_index(drop=True)


# ---- target formulations ------------------------------------------------------
# kind -> (training target, sample weight, multiplier back to modelled physical quantity)
def target_spec(d, kind, test=False):
    A = d.At if test else d.A
    M = d.Mt if test else d.M
    if kind.startswith(("gas_per:", "elec_per:", "lgas_per:", "lelec_per:")):
        # ratio formulations: target / D(col), weight D^2 for linear kinds (loss == target-scale MSE)
        fam, col = kind.split(":")
        D = (d.Xt if test else d.X)[col].values.astype(float)
        to_phys = D / A if fam in ("gas_per", "lgas_per") else D  # gas family stores EUI
        if test:
            return None, None, to_phys
        tgt = d.y[TARGET_COLS[0]] if fam in ("gas_per", "lgas_per") else d.y[TARGET_COLS[1]]
        if fam.startswith("l"):
            return np.log(tgt / D), None, to_phys
        return tgt / D, D ** 2 / np.mean(D ** 2), to_phys
    if test:
        return None, None, {"eui": 1.0, "gas": 1.0 / A, "gpm": M / A, "leui": 1.0,
                            "elec": 1.0, "epm": M, "epa": A, "lepm": M}.get(kind)
    y_gas, y_elec, y_eui = (d.y[c] for c in TARGET_COLS[:3])
    if kind == "eui":   # model EUI directly
        return y_eui, None, 1.0
    if kind == "gas":   # model gas, convert to EUI
        return y_gas, None, 1.0 / A
    if kind == "gpm":   # gas per meter, weighted so loss equals gas MSE
        return y_gas / M, M ** 2 / np.mean(M ** 2), M / A
    if kind == "leui":  # log EUI; prediction is exp()-ed (blend weights absorb the retransform bias)
        return np.log(y_eui), None, 1.0
    if kind == "elec":
        return y_elec, None, 1.0
    if kind == "lepm":
        return np.log(y_elec / M), None, M
    if kind == "epm":
        return y_elec / M, M ** 2 / np.mean(M ** 2), M
    if kind == "epa":
        return y_elec / A, A ** 2 / np.mean(A ** 2), A
    raise ValueError(kind)


# ---- model back-ends ------------------------------------------------------------
def fit_xgb(params, Xa, ya, wa, Xb, yb, wb, Xt):
    import xgboost as xgb
    p = dict(params)
    n = p.pop("n_estimators", 10000)
    es = p.pop("early_stopping_rounds", 200)
    p.setdefault("device", "cuda")
    p.setdefault("tree_method", "hist")
    p.setdefault("seed", SEED)
    mb = p.get("max_bin", 256)  # the quantile sketch must use the booster's bin count
    da = xgb.QuantileDMatrix(Xa, ya, weight=wa, max_bin=mb)
    db = xgb.QuantileDMatrix(Xb, yb, weight=wb, ref=da, max_bin=mb)
    m = xgb.train(p, da, n, evals=[(db, "va")], early_stopping_rounds=es, verbose_eval=False)
    rng = (0, m.best_iteration + 1)
    return m.inplace_predict(Xb, iteration_range=rng), m.inplace_predict(Xt, iteration_range=rng), m.best_iteration


def fit_lgb(params, Xa, ya, wa, Xb, yb, wb, Xt):
    import lightgbm as lgb
    p = dict(objective="regression", seed=SEED, deterministic=True, force_row_wise=True, num_threads=16, verbose=-1)
    p.update(params)
    n = p.pop("n_estimators", 10000)
    es = p.pop("early_stopping_rounds", 200)
    da = lgb.Dataset(Xa, ya, weight=wa, free_raw_data=False)
    db = lgb.Dataset(Xb, yb, weight=wb, reference=da)
    m = lgb.train(p, da, n, valid_sets=[db], callbacks=[lgb.early_stopping(es, verbose=False)])
    return m.predict(Xb, num_iteration=m.best_iteration), m.predict(Xt, num_iteration=m.best_iteration), m.best_iteration


def fit_cat(params, Xa, ya, wa, Xb, yb, wb, Xt):
    from catboost import CatBoostRegressor, Pool
    p = dict(loss_function="RMSE", task_type="GPU", random_seed=SEED, verbose=0, iterations=20000,
             od_type="Iter", od_wait=300, use_best_model=True, gpu_ram_part=0.45)
    p.update(params)
    m = CatBoostRegressor(**p)
    m.fit(Pool(Xa, ya, weight=wa), eval_set=Pool(Xb, yb, weight=wb))
    return m.predict(Xb), m.predict(Xt), m.get_best_iteration()


BACKENDS = {"xgb": fit_xgb, "lgb": fit_lgb, "cat": fit_cat}
LOG_KINDS = ("leui", "lepm")


def is_log_kind(kind):
    return kind in LOG_KINDS or kind.startswith(("lgas_per:", "lelec_per:"))


def is_gas_kind(kind):
    return kind in ("eui", "gas", "gpm", "leui") or kind.startswith(("gas_per:", "lgas_per:"))


def run(d, name, backend, kind, params, use_te=False, windows=(), drop=(), out_dir=".", verbose=True,
        wmul=None, extra=None, extra_t=None, cells=False, te_full=False, knn=False):
    """K-fold CV; saves OOF and fold-averaged test predictions of the modelled physical quantity.

    The saved quantity is EUI for gas-family kinds and elec for elec-family kinds.
    te_full: encode validation rows from all other train rows (leave-one-out) and test rows from all
    train rows, instead of from the fold's fitting part only (training rows keep inner-OOF encodings).
    """
    fit = BACKENDS[backend]
    y, w, mult = target_spec(d, kind)
    mult_t = target_spec(d, kind, test=True)[2]
    if wmul is not None:  # extra per-row training weight (e.g. inverse noise variance)
        w = wmul / wmul.mean() if w is None else w * wmul / np.mean(w * wmul)
    oof = np.zeros(len(d.X))
    pt = np.zeros(len(d.Xt))
    its = []
    t0 = time.time()
    for k, (a, b) in enumerate(d.folds):
        Xa, Xb, Xt = d.X.iloc[a], d.X.iloc[b], d.Xt
        if extra is not None:  # additional precomputed feature frames (train rows / test rows)
            Xa = pd.concat([Xa.reset_index(drop=True), extra.iloc[a].reset_index(drop=True)], axis=1)
            Xb = pd.concat([Xb.reset_index(drop=True), extra.iloc[b].reset_index(drop=True)], axis=1)
            Xt = pd.concat([Xt.reset_index(drop=True), extra_t.reset_index(drop=True)], axis=1)
        if use_te:
            Xa = pd.concat([Xa.reset_index(drop=True), d.te_features(a, "oof", windows, cells)], axis=1)
            if te_full:
                if k == 0:
                    allr = np.arange(len(d.X))
                    Tt_full = d.te_features(allr, d.te_apply(), windows, cells)
                Tb = d.te_features(allr, d.te_apply(b), windows, cells, self_idx=b)
                Tt = Tt_full
            else:
                Tb = d.te_features(a, d.te_apply(b), windows, cells)
                Tt = d.te_features(a, d.te_apply(), windows, cells)
            Xb = pd.concat([Xb.reset_index(drop=True), Tb], axis=1)
            Xt = pd.concat([Xt.reset_index(drop=True), Tt], axis=1)
        if knn:  # FB07 kNN target encodings, cross-fitted like the target encodings
            Xa = pd.concat([Xa.reset_index(drop=True), d.knn_features(a, "oof")], axis=1)
            Xb = pd.concat([Xb.reset_index(drop=True), d.knn_features(a, ("train", b))], axis=1)
            Xt = pd.concat([Xt.reset_index(drop=True), d.knn_features(a, "test")], axis=1)
        if drop:
            Xa, Xb, Xt = (z.drop(columns=list(drop)) for z in (Xa, Xb, Xt))
        pb, ptk, it = fit(params, Xa, y[a], None if w is None else w[a], Xb, y[b], None if w is None else w[b], Xt)
        if is_log_kind(kind):
            pb, ptk = np.exp(pb), np.exp(ptk)
        oof[b] = pb * (mult if np.isscalar(mult) else mult[b])
        pt += ptk * mult_t / len(d.folds)
        its.append(it)
        if verbose:
            print(f"  [{name}] fold {k} it={it} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{out_dir}/oof", exist_ok=True)
    os.makedirs(f"{out_dir}/pred", exist_ok=True)
    np.save(f"{out_dir}/oof/{name}.npy", oof)
    np.save(f"{out_dir}/pred/{name}.npy", pt)
    if is_gas_kind(kind):
        msg = (f"eui={nrmse(d.y[TARGET_COLS[2]], oof, TARGET_COLS[2]):.5f} "
               f"gas={nrmse(d.y[TARGET_COLS[0]], oof * d.A, TARGET_COLS[0]):.5f}")
    else:
        msg = f"elec={nrmse(d.y[TARGET_COLS[1]], oof, TARGET_COLS[1]):.5f}"
    print(f"[{name}] {msg} iters={its} time={time.time() - t0:.0f}s", flush=True)
    return oof, pt
