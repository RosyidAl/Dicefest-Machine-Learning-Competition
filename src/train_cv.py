"""Config-driven CV runner (MASTER_PROMPT 3.2).

usage: python -m src.train_cv --config configs/E0200.yaml [--smoke]
       (or python src/train_cv.py ...)

For every target and fold: build fold-aware features, transform the target, fit, predict validation and test rows,
inverse-transform, clip. Writes experiments/<exp_id>_<slug>/{config.yaml, run.log, metrics.json, oof.parquet,
test_pred.parquet, feature_importance.csv, env.json, README.md} and a registry row (running -> done / failed).

Config keys (see configs/*.yaml): exp_id, parent_id, phase, hypothesis, slug, targets, cv {scheme, fold_file,
screen_folds}, features {set, blocks}, target_transform {per target or default}, model {family, device, params},
seeds, subsample, postprocess {clip}.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import base_features, constant_cols, load  # noqa: E402
from metric import OFFICIAL_SIGMAS, TARGET_COLS  # noqa: E402
from seed import seed_everything  # noqa: E402
from targets import Transform  # noqa: E402
import tracking  # noqa: E402

SHORT = {"t1": TARGET_COLS[0], "t2": TARGET_COLS[1], "t3": TARGET_COLS[2], "t4": TARGET_COLS[3], "t5": TARGET_COLS[4]}
ENGINEERED = ["zone_n", "sub_n", "hvac_rest", "area_per_meter", "area_per_epc", "gea_ratio", "a07_ratio", "a08_ratio",
              "a05_ratio", "a04_ratio", "s002_ratio", "meters_per_epc", "clean_ratio"]


# ---------------------------------------------------------------- features
class FeatureBuilder:
    """Feature sets:
    raw      201 original features (2 string columns label-encoded), no spatial ids
    raw_ids  raw + zone / subzone ids as numbers
    legacy   FS_legacy: raw_ids + physics ratios (common.base_features); blocks add more
    blocks   agg (transductive subzone/zone aggregates), te (fold-aware OOF target encodings: subzone, zone,
             sub_n windows 2/10, climate cells), plus Phase 4 blocks registered in src/features/
    """

    def __init__(self, cfg, tr, te):
        self.set = cfg.get("set", "legacy")
        self.blocks = list(cfg.get("blocks", []))
        const = constant_cols(tr)
        X, Xt = base_features(tr, const), base_features(te, const)
        if self.set in ("raw", "raw_ids"):
            keep = [c for c in X.columns if c not in ENGINEERED or (self.set == "raw_ids" and c in ("zone_n", "sub_n"))]
            X, Xt = X[keep], Xt[keep]
        self.X, self.Xt = X.reset_index(drop=True), Xt.reset_index(drop=True)
        self.d = None
        if "te" in self.blocks or "agg" in self.blocks:
            from train_lib import Data
            self.d = Data()
        if "agg" in self.blocks:
            ex, ext = self.d.agg_features()
            self.X, self.Xt = pd.concat([self.X, ex], axis=1), pd.concat([self.Xt, ext], axis=1)
        self.extra_fold_blocks = []
        for b in self.blocks:
            if b not in ("te", "agg"):
                from features import BLOCKS
                blk = BLOCKS[b](tr, te, **cfg.get("block_params", {}).get(b, {}))
                if blk.level == "transductive":
                    A, At = blk.transform()
                    self.X, self.Xt = pd.concat([self.X, A], axis=1), pd.concat([self.Xt, At], axis=1)
                else:
                    self.extra_fold_blocks.append(blk)

    def fold(self, a, b):
        Xa, Xb, Xt = self.X.iloc[a].reset_index(drop=True), self.X.iloc[b].reset_index(drop=True), self.Xt
        if "te" in self.blocks:
            d = self.d
            w, c = (2, 10), True
            Xa = pd.concat([Xa, d.te_features(a, "oof", w, c)], axis=1)
            Xb = pd.concat([Xb, d.te_features(a, d.te_apply(b), w, c)], axis=1)
            Xt = pd.concat([Xt, d.te_features(a, d.te_apply(), w, c)], axis=1)
        for blk in self.extra_fold_blocks:
            Fa, Fb, Ft = blk.fold(a, b)
            Xa, Xb, Xt = pd.concat([Xa, Fa], axis=1), pd.concat([Xb, Fb], axis=1), pd.concat([Xt, Ft], axis=1)
        return Xa, Xb, Xt


# ---------------------------------------------------------------- models
def fit_predict(family, params, Xa, ya, wa, Xb, yb, wb, Xt, seed, cat_cols=()):
    """returns (pred_b, pred_t, best_iteration, importance Series or None)"""
    p = dict(params)
    if family == "lgb":
        import lightgbm as lgb
        q = dict(objective="regression", seed=seed, deterministic=True, force_row_wise=True, num_threads=8, verbose=-1)
        q.update(p)
        n, es = q.pop("n_estimators", 20000), q.pop("early_stopping_rounds", 300)
        da = lgb.Dataset(Xa, ya, weight=wa, free_raw_data=False, categorical_feature=list(cat_cols) or "auto")
        db = lgb.Dataset(Xb, yb, weight=wb, reference=da)
        m = lgb.train(q, da, n, valid_sets=[db], callbacks=[lgb.early_stopping(es, verbose=False)])
        imp = pd.Series(m.feature_importance("gain"), Xa.columns)
        return m.predict(Xb, num_iteration=m.best_iteration), m.predict(Xt, num_iteration=m.best_iteration), \
            m.best_iteration, imp
    if family == "xgb":
        import xgboost as xgb
        n, es = p.pop("n_estimators", 20000), p.pop("early_stopping_rounds", 300)
        p.setdefault("device", "cuda")
        p.setdefault("tree_method", "hist")
        p["seed"] = seed
        mb = p.get("max_bin", 256)
        cat = bool(p.pop("enable_categorical", False))
        da = xgb.QuantileDMatrix(Xa, ya, weight=wa, max_bin=mb, enable_categorical=cat)
        db = xgb.QuantileDMatrix(Xb, yb, weight=wb, ref=da, max_bin=mb, enable_categorical=cat)
        m = xgb.train(p, da, n, evals=[(db, "va")], early_stopping_rounds=es, verbose_eval=False)
        r = (0, m.best_iteration + 1)
        dbp = xgb.DMatrix(Xb, enable_categorical=cat)
        dtp = xgb.DMatrix(Xt, enable_categorical=cat)
        sc = m.get_score(importance_type="total_gain")
        imp = pd.Series({c: sc.get(c, 0.0) for c in Xa.columns})
        return m.predict(dbp, iteration_range=r), m.predict(dtp, iteration_range=r), m.best_iteration, imp
    if family == "cat":
        from catboost import CatBoostRegressor, Pool
        q = dict(loss_function="RMSE", task_type="GPU", random_seed=seed, verbose=0, iterations=20000, od_type="Iter",
                 od_wait=300, use_best_model=True, gpu_ram_part=0.25)
        q.update(p)
        m = CatBoostRegressor(**q)
        m.fit(Pool(Xa, ya, weight=wa, cat_features=list(cat_cols) or None),
              eval_set=Pool(Xb, yb, weight=wb, cat_features=list(cat_cols) or None))
        imp = pd.Series(m.get_feature_importance(), Xa.columns)
        return m.predict(Pool(Xb, cat_features=list(cat_cols) or None)), \
            m.predict(Pool(Xt, cat_features=list(cat_cols) or None)), m.get_best_iteration(), imp
    if family == "mean":
        mu = float(np.average(ya, weights=wa))
        return np.full(len(Xb), mu), np.full(len(Xt), mu), 0, None
    if family in ("ridge", "knn", "extratrees"):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Za, Zb, Zt = (np.nan_to_num(z.values.astype(np.float64)) for z in (Xa, Xb, Xt))
        if p.pop("log1p_skewed", False):
            sk = [j for j in range(Za.shape[1]) if Za[:, j].min() >= 0 and np.percentile(Za[:, j], 99) >
                  5 * max(np.median(Za[:, j]), 1e-6)]
            for z in (Za, Zb, Zt):
                z[:, sk] = np.log1p(z[:, sk])
        if family == "ridge":
            from sklearn.linear_model import Ridge
            m = make_pipeline(StandardScaler(), Ridge(alpha=p.get("alpha", 1.0)))
            m.fit(Za, ya, ridge__sample_weight=wa)
            imp = pd.Series(np.abs(m[-1].coef_), Xa.columns)
        elif family == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            m = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=p.get("k", 20), weights="distance",
                                                                     n_jobs=p.get("n_jobs", 8)))
            m.fit(Za, ya)
            imp = None
        else:
            from sklearn.ensemble import ExtraTreesRegressor
            m = ExtraTreesRegressor(random_state=seed, **{k: v for k, v in p.items()})
            m.fit(Za, ya, sample_weight=wa)
            imp = pd.Series(m.feature_importances_, Xa.columns)
        return m.predict(Zb), m.predict(Zt), 0, imp
    raise ValueError(family)


# ---------------------------------------------------------------- runner
def main(cfg_path, smoke=False):
    cfg = yaml.safe_load(open(cfg_path))
    eid, slug = cfg["exp_id"], cfg.get("slug", "exp")
    if smoke:
        eid = eid + "_smoke"
    out = f"experiments/{eid}_{slug}"
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/run.log", "a")

    def log(msg):
        line = f"[{tracking.now_wib()}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"), sort_keys=False)
    json.dump(tracking.env_info(), open(f"{out}/env.json", "w"), indent=1)
    targets = [SHORT.get(t, t) for t in cfg["targets"]]
    mcfg = cfg["model"]
    fam = mcfg["family"]
    cvc = cfg.get("cv", {})
    fold_file = cvc.get("fold_file", "data/folds/folds_primary_k5_s42.parquet")
    seeds = cfg.get("seeds", [42])
    row = dict(exp_id=eid, parent_id=cfg.get("parent_id", ""), status="running", phase=cfg.get("phase", ""),
               created_wib=tracking.now_wib(), hypothesis=cfg.get("hypothesis", ""), model_family=fam,
               device=mcfg.get("device", "cpu"), targets=",".join(cfg["targets"]),
               target_transform=json.dumps(cfg.get("target_transform", {}), sort_keys=True),
               feature_set_id=cfg.get("features", {}).get("set", "legacy") + "+" + "+".join(
                   cfg.get("features", {}).get("blocks", [])), cv_scheme=cvc.get("scheme", "primary"),
               fold_file=fold_file, seeds=",".join(map(str, seeds)), config_path=cfg_path,
               params_hash=tracking.params_hash(cfg), git_commit=tracking.git_commit(), data_hash=tracking.data_hash())
    if not smoke:
        tracking.upsert(row)
    t0 = time.time()
    peak = [0.0]
    try:
        seed_everything(seeds[0])
        tr, te = load()
        folds = pd.read_parquet(fold_file)
        assert (folds["id"].values == tr["id"].values).all()
        fold = folds["fold"].values
        nf = int(fold.max()) + 1
        use_folds = list(range(cvc.get("screen_folds", nf)))
        sub = cfg.get("subsample")
        if smoke:
            use_folds, sub = use_folds[:2], 0.1
            key = {"lgb": "n_estimators", "xgb": "n_estimators", "cat": "iterations"}.get(fam)
            if key:
                mcfg.setdefault("params", {})[key] = min(mcfg.get("params", {}).get(key, 300), 300)
        fit_mask = np.ones(len(tr), bool)
        if sub:
            fit_mask = np.random.default_rng(42).random(len(tr)) < sub
        fb = FeatureBuilder(cfg.get("features", {}), tr, te)
        log(f"features: {fb.X.shape[1]} base columns, blocks {fb.blocks}; folds {use_folds} of {nf}; "
            f"subsample {sub}")
        ctx_cols = ["bld_dim_06", "bld_dim_12", "bld_dim_01", "bld_dim_03"]
        ctx_tr = {c: tr[c].values for c in ctx_cols}
        ctx_te = {c: te[c].values for c in ctx_cols}
        tcfg = cfg.get("target_transform", {})
        oof = pd.DataFrame({"id": tr["id"].values})
        tp = pd.DataFrame({"id": te["id"].values})
        imp_all = []
        metrics = {"exp_id": eid, "cv_scheme": row["cv_scheme"], "fold_file": fold_file, "per_target": {}}
        eval_mask = np.isin(fold, use_folds) & fit_mask
        cat_cols = mcfg.get("cat_cols", [])
        multi = mcfg.get("multi_output", False)
        cache = {}
        if fam == "mlp_multi":  # one network for all targets (B6): standardised targets, fixed final epoch
            from nn import fit_fold, preprocess, set_seed
            mp = dict(mcfg.get("params", {}))
            E = mp.pop("epochs", 20)
            if smoke:
                E = 2
            for k in use_folds:
                a = np.flatnonzero((fold != k) & fit_mask)
                b = np.flatnonzero((fold == k) & fit_mask)
                Xa, Xb, Xt = fb.fold(a, b)
                Za, Zt_ = preprocess(pd.concat([Xa, Xb]), Xt)
                Za, Zb = Za[:len(a)], Za[len(a):]
                Ts = [Transform(**tcfg.get(t, tcfg.get("default", {"type": "identity"}))) for t in targets]
                Ya = np.column_stack([T.fit(tr[t].values[a], None).forward(tr[t].values[a], None)[0]
                                      for T, t in zip(Ts, targets)])
                Yb = np.column_stack([T.forward(tr[t].values[b], None)[0] for T, t in zip(Ts, targets)])
                pb_s, pt_s = np.zeros((len(b), len(targets))), np.zeros((len(te), len(targets)))
                for s in seeds:
                    set_seed(s + k)
                    pb, ptk, _ = fit_fold(Za, Ya, np.ones_like(Ya), Zb, Yb, np.ones_like(Yb), Zt_, epochs=E,
                                          fixed_ep=E - 1, **mp)
                    pb_s += pb / len(seeds)
                    pt_s += ptk / len(seeds)
                for j, (T, t) in enumerate(zip(Ts, targets)):
                    cache[(t, k)] = (T.inverse(pb_s[:, j], None), T.inverse(pt_s[:, j], None))
                log(f"mlp_multi fold {k} done ({time.time() - t0:.0f}s)")
        for tgt in targets:
            tc = tcfg.get(tgt, tcfg.get("default", {"type": "identity"}))
            o = np.full(len(tr), np.nan)
            pt = np.zeros(len(te))
            fold_scores, its = [], []
            for k in use_folds:
                a = np.flatnonzero((fold != k) & fit_mask)
                b = np.flatnonzero((fold == k) & fit_mask)
                if cache:
                    o[b], pt_k = cache[(tgt, k)]
                    pt += pt_k / len(use_folds)
                    y = tr[tgt].values
                    fs = float(np.sqrt(np.mean((o[b] - y[b]) ** 2)) / OFFICIAL_SIGMAS[tgt])
                    fold_scores.append(fs)
                    its.append(0)
                    continue
                Xa, Xb, Xt = fb.fold(a, b)
                if cat_cols:
                    for c in cat_cols:
                        for z in (Xa, Xb, Xt):
                            z[c] = z[c].astype(int)
                y = tr[tgt].values
                T = Transform(**tc).fit(y[a], {c: v[a] for c, v in ctx_tr.items()})
                ya, wa = T.forward(y[a], {c: v[a] for c, v in ctx_tr.items()})
                yb, wb = T.forward(y[b], {c: v[b] for c, v in ctx_tr.items()})
                pb_s, pt_s = np.zeros(len(b)), np.zeros(len(te))
                for s in seeds:
                    pb, ptk, it, imp = fit_predict(fam, mcfg.get("params", {}), Xa, ya, wa, Xb, yb, wb, Xt, s,
                                                   cat_cols)
                    pb_s += T.inverse(np.asarray(pb, float), {c: v[b] for c, v in ctx_tr.items()}) / len(seeds)
                    pt_s += T.inverse(np.asarray(ptk, float), ctx_te) / len(seeds)
                    its.append(int(it))
                    if imp is not None:
                        imp_all.append(imp.rename(f"{tgt}_f{k}_s{s}"))
                o[b] = pb_s
                pt += pt_s / len(use_folds)
                fs = float(np.sqrt(np.mean((o[b] - y[b]) ** 2)) / OFFICIAL_SIGMAS[tgt])
                fold_scores.append(fs)
                peak[0] = max(peak[0], psutil.Process().memory_info().rss / 2 ** 30)
                log(f"{tgt} fold {k}: nrmse {fs:.5f} it {its[-1]} ({time.time() - t0:.0f}s)")
            lo, hi = (0, 100) if tgt in TARGET_COLS[3:] else (0, None)
            if cfg.get("postprocess", {}).get("clip", True):
                o = np.where(np.isnan(o), o, np.clip(o, lo, hi))
                pt = np.clip(pt, lo, hi)
            m = eval_mask
            score = float(np.sqrt(np.mean((o[m] - tr[tgt].values[m]) ** 2)) / OFFICIAL_SIGMAS[tgt])
            metrics["per_target"][tgt] = {"nrmse": score, "rmse": score * OFFICIAL_SIGMAS[tgt],
                                          "fold_nrmse": fold_scores, "best_iteration": its}
            oof[tgt], tp[tgt] = o, pt
            log(f"{tgt}: OOF nrmse {score:.5f}")
            if tgt == TARGET_COLS[0] and cfg.get("derive_eui"):  # EUI = gas / floor area (exact identity)
                t3 = TARGET_COLS[2]
                o3, Av = o / tr["bld_dim_06"].values, tr["bld_dim_06"].values
                y3 = tr[t3].values
                s3 = float(np.sqrt(np.mean((o3[m] - y3[m]) ** 2)) / OFFICIAL_SIGMAS[t3])
                f3 = [float(np.sqrt(np.mean((o3[(fold == k) & fit_mask] - y3[(fold == k) & fit_mask]) ** 2))
                            / OFFICIAL_SIGMAS[t3]) for k in use_folds]
                metrics["per_target"][t3] = {"nrmse": s3, "rmse": s3 * OFFICIAL_SIGMAS[t3], "fold_nrmse": f3,
                                             "best_iteration": its, "derived_from": TARGET_COLS[0]}
                oof[t3], tp[t3] = o3, pt / te["bld_dim_06"].values
                log(f"{t3} (derived gas / A): OOF nrmse {s3:.5f}")
        if len(targets) == 5:
            metrics["nrmse_mean"] = float(np.mean([metrics["per_target"][t]["nrmse"] for t in TARGET_COLS]))
        else:
            metrics["nrmse_mean"] = None
        metrics["runtime_min"] = (time.time() - t0) / 60
        metrics["peak_ram_gb"] = peak[0]
        try:
            import torch
            metrics["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
        except ImportError:
            metrics["peak_vram_gb"] = 0
        json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)
        oof.to_parquet(f"{out}/oof.parquet", index=False)
        tp.to_parquet(f"{out}/test_pred.parquet", index=False)
        if imp_all:
            pd.concat(imp_all, axis=1).fillna(0).assign(mean=lambda z: z.mean(1)).sort_values(
                "mean", ascending=False).to_csv(f"{out}/feature_importance.csv")
        targets = [t for t in TARGET_COLS if t in metrics["per_target"]]
        idx = {t: f"nrmse_t{TARGET_COLS.index(t) + 1}" for t in targets}
        row.update(status="done", finished_wib=tracking.now_wib(), n_features=fb.X.shape[1],
                   runtime_min=f"{metrics['runtime_min']:.1f}", peak_ram_gb=f"{peak[0]:.1f}",
                   peak_vram_gb=f"{metrics['peak_vram_gb']:.1f}", oof_path=f"{out}/oof.parquet",
                   test_pred_path=f"{out}/test_pred.parquet",
                   fold_nrmse_json=json.dumps({t: [round(x, 5) for x in metrics["per_target"][t]["fold_nrmse"]]
                                               for t in targets}),
                   nrmse_mean="" if metrics["nrmse_mean"] is None else f"{metrics['nrmse_mean']:.5f}",
                   notes=cfg.get("notes", "") + (f" screen_folds={len(use_folds)}" if len(use_folds) < nf else "")
                   + (f" subsample={sub}" if sub else ""))
        for t in targets:
            row[idx[t]] = f"{metrics['per_target'][t]['nrmse']:.5f}"
        with open(f"{out}/README.md", "w") as f:
            f.write(f"# {eid} {slug}\n\n- Hipotesis: {cfg.get('hypothesis', '')}\n- Setup: model {fam}, fitur "
                    f"{row['feature_set_id']}, transform {row['target_transform']}, CV {fold_file} folds {use_folds}, "
                    f"seed {seeds}\n- Hasil (NRMSE OOF):\n")
            for t in targets:
                f.write(f"  - {t}: {metrics['per_target'][t]['nrmse']:.5f} (fold {np.round(fold_scores, 5).tolist() if t == targets[-1] else ''})\n")
            f.write(f"- Runtime {metrics['runtime_min']:.1f} menit, peak RAM {peak[0]:.1f} GB\n- Kesimpulan: lihat reports/JOURNAL.md\n")
        log(f"done in {metrics['runtime_min']:.1f} min")
    except Exception:  # noqa: BLE001
        log("FAILED\n" + traceback.format_exc())
        row.update(status="failed", finished_wib=tracking.now_wib(), notes="see run.log")
        if not smoke:
            tracking.upsert(row)
        raise
    if not smoke:
        tracking.upsert(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.config, a.smoke)
