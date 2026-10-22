"""Final DICE FEST 2026 pipeline: data -> 50 cross-validated models -> NNLS blend -> adoption model -> residual
correction -> train-duplicate copy -> submission.

Stages (all deterministic, seeds fixed in the configs):
1. members  : every model of configs/members.yaml, 5- or 10-fold KFold(shuffle, random_state=42) over train rows,
              saving OOF and fold-averaged test predictions to <work>/oof|pred/<name>.npy. Two lanes run in parallel:
              'gpu' (XGBoost CUDA, PyTorch MLPs) and 'cpu' (LightGBM).
2. adoption : electric-heating adoption (exact household counts, LightGBM for unseen subzones), adopt_v2.py.
3. assemble : non-negative least-squares blend per scored column on the OOF predictions, clean-building ratio from
              its exact identity, residual correction with frozen weights (configs/resid_knn.json), copy of the train
              targets for the 77 test rows that duplicate a train row, clip, write CSV + sha256.
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SEED, SIGMAS, TARGET_COLS, load, load_sample, nrmse  # noqa: E402

CONF = os.path.join(HERE, "..", "configs")
ELEC = ("epm", "elec", "lepm", "epa", "lepa")
GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
METRICS = {"gas": GAS, "eui": EUI, "elec": EL}


def load_config():
    return yaml.safe_load(open(os.path.join(CONF, "members.yaml")))


def _wmul(d, code):
    return {None: None, "iA": 1.0 / d.A, "iM": 1.0 / d.M, "A": d.A, "M": d.M.astype(float),
            "sqrtM": np.sqrt(d.M)}[code]


LANES = ("xgb", "nn", "cpu")  # parallel worker processes: XGBoost (GPU), MLPs (GPU), LightGBM (CPU)
LANE_THREADS = {"xgb": "2", "nn": "2", "cpu": "12", "gpu": "4"}


def lane_of(spec, lane):
    """does the member belong to `lane`? ('gpu' = xgb + nn, kept for a two-lane run)"""
    b = spec["backend"]
    kind = "xgb" if b == "xgb" else ("nn" if b in ("nn", "nn_multi", "nca") else "cpu")
    return kind == lane or (lane == "gpu" and kind in ("xgb", "nn"))


def make_data(work):
    """Data for the members: train_lib.Data plus the FB07b kNN block (fb07b.DataX), which is switched on per member
    (spec knn_x: true) and off otherwise, so FB07-only members see exactly the train_lib.Data features"""
    from fb07b import DataX
    from postprocess import spaces
    d = DataX()
    tr, te = load()
    S = spaces(tr, te)
    d.sp = {k: (v[:len(tr)], v[len(tr):]) for k, v in S.items()}
    d.use_x = False
    d.knn_cache_dir = f"{work}/knn_cache"  # FB07 fold encodings shared between members and lanes (same values)
    d.knn_x_cache_dir = f"{work}/knn_x_cache"  # FB07b fold encodings
    return d


def run_members(lane, work, only=None):
    """train every member of a lane (in config order); members with existing outputs are skipped"""
    from nn import run_nn, run_nn_multi
    from train_lib import Data, run
    cfg = load_config()
    os.makedirs(f"{work}/oof", exist_ok=True)
    os.makedirs(f"{work}/pred", exist_ok=True)
    d = make_data(work)
    ex, ext = d.agg_features()
    done_multi = set()
    for name, spec in cfg["members"].items():
        if not lane_of(spec, lane) or (only and name not in only):
            continue
        if spec["backend"] == "nn_multi":
            g = spec["group"]
            if g in done_multi:
                continue
            done_multi.add(g)
            m = cfg["multi"][g]
            outs = [f"{work}/pred/{g}_{t[0]}.npy" for t in m["tasks"]]
            if all(os.path.exists(o) for o in outs):
                continue
            d.folds = list(KFold(m["n_folds"], shuffle=True, random_state=SEED).split(d.X))
            d.use_x = bool(m.get("knn_x", False))
            tasks = [(s, k, _wmul(d, w)) for s, k, w in m["tasks"]]
            agg = dict(extra=ex, extra_t=ext) if m["agg"] else {}
            run_nn_multi(d, g, tasks, out_dir=work, use_te=m["use_te"], windows=tuple(m["windows"]), cells=m["cells"],
                         knn=m.get("knn", False), epochs=m["epochs"], fixed_ep=m["fixed_ep"], seeds=tuple(m["seeds"]),
                         model_dir=f"{work}/models/{g}", **agg, **m["params"])
            continue
        if os.path.exists(f"{work}/pred/{name}.npy"):
            continue
        d.folds = list(KFold(spec["n_folds"], shuffle=True, random_state=SEED).split(d.X))
        d.use_x = bool(spec.get("knn_x", False))
        agg = dict(extra=ex, extra_t=ext) if spec["agg"] else {}
        te = dict(use_te=spec.get("use_te", False), windows=tuple(spec.get("windows", ())),
                  cells=spec.get("cells", False), knn=spec.get("knn", False))
        if spec["backend"] == "nca":  # retrieval regressor (NCA0001, nca.py)
            from nca import run_nca
            run_nca(d, name, spec["kind"], out_dir=work, wmul=_wmul(d, spec["wmul"]), seeds=tuple(spec["seeds"]),
                    model_dir=f"{work}/models/{name}", **te, **agg, **dict(spec["params"]))
        elif spec["backend"] == "nn":
            p = dict(spec["params"])
            p["hidden"] = tuple(p["hidden"])
            run_nn(d, name, spec["kind"], out_dir=work, wmul=_wmul(d, spec["wmul"]), epochs=spec["epochs"],
                   fixed_ep=spec["fixed_ep"], seeds=tuple(spec["seeds"]), model_dir=f"{work}/models/{name}", **te,
                   **agg, **p)
        else:
            run(d, name, spec["backend"], spec["kind"], dict(spec["params"]), out_dir=work,
                wmul=_wmul(d, spec["wmul"]), model_dir=f"{work}/models/{name}", **te, **agg)


def run_adoption(work):
    from adopt_v2 import fit_adoption_v2
    pred, rmse_unseen, frac_unseen = fit_adoption_v2()
    np.save(f"{work}/pred/adopt_v2.npy", pred)
    return {"adoption_unseen_weighted_rmse": rmse_unseen, "unseen_fraction": frac_unseen}


def _correction(args):
    """residual-correction features of one metric (run in a worker process)"""
    from postprocess import all_features
    from threadpoolctl import threadpool_limits
    metric, kt, ke, r, pf, pa, A, At, Zs, n, n_te, subn, subnt, lam, configs = args
    with threadpool_limits(1):  # single-threaded BLAS: multithreaded gemv changes the last bit with the thread count
        F = all_features(metric, kt, ke, r, pf, pa, A, At, Zs, np.arange(n), np.arange(n, n + n_te), subn, subnt,
                         configs)
        return metric, F @ np.asarray(lam)


def assemble(work, out_csv, log=print):
    """blend + adoption + residual correction + duplicate copy -> submission; BLAS is limited to one thread so that
    the output bytes do not depend on the machine's core count (OpenBLAS gemv sums in a thread-dependent order)"""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _assemble(work, out_csv, log)


def _assemble(work, out_csv, log=print):
    from blend import cv_blend_score, nnls_blend
    from dup_fix import duplicate_targets
    from postprocess import KSHR, blend_preds, group_keys, spaces
    cfg = load_config()
    tr, te = load()
    A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
    y = {c: tr[c].values for c in TARGET_COLS}
    runs = {"gas": cfg["blend"]["gas"], "elec": cfg["blend"]["elec"]}
    P = {f: (np.column_stack([np.load(f"{work}/oof/{r}.npy") for r in runs[f]]),
             np.column_stack([np.load(f"{work}/pred/{r}.npy") for r in runs[f]])) for f in runs}
    summary = {}
    lamcfg = json.load(open(os.path.join(CONF, "resid_knn.json")))
    vc = lamcfg.get("vc", {})  # metric -> varying-coefficient spec (D012); empty = one global weight vector
    M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
    # 1. blend (as src/make_submission.py) and its honest nested-CV score
    w = {"eui": nnls_blend(P["gas"][0], y[EUI]), "gas": nnls_blend(P["gas"][0], y[EUI], row_scale=A),
         "elec": nnls_blend(P["elec"][0], y[EL])}
    summary["weights"] = {m: dict(zip(runs["elec" if m == "elec" else "gas"], np.round(w[m], 6).tolist())) for m in w}
    summary["cv_blend_honest"] = {
        "gas": nrmse(y[GAS], cv_blend_score(P["gas"][0], y[EUI], A) * A, GAS),
        "eui": nrmse(y[EUI], cv_blend_score(P["gas"][0], y[EUI]), EUI),
        "elec": nrmse(y[EL], cv_blend_score(P["elec"][0], y[EL]), EL)}
    log(f"blend honest CV {summary['cv_blend_honest']}")
    sub = pd.DataFrame({"id": te["id"].values})
    sub[GAS] = np.clip(P["gas"][1] @ w["gas"], 0, None) * At
    sub[EL] = np.clip(P["elec"][1] @ w["elec"], 0, None)
    sub[EUI] = np.clip(P["gas"][1] @ w["eui"], 0, None)
    sub[TARGET_COLS[3]] = np.round(100 * te["bld_dim_02"].values / te["bld_dim_01"].values, 2)
    sub[TARGET_COLS[4]] = np.clip(np.load(f"{work}/pred/adopt_v2.npy"), 0, 100)
    if vc:  # member weights piecewise linear in the ranks of A, M and the global blend prediction (blend.blend_vc)
        for metric, col in METRICS.items():
            fam = "elec" if metric == "elec" else "gas"
            _, pa = blend_preds(P[fam][0], P[fam][1], y[EL if metric == "elec" else EUI], metric, A, At,
                                vc.get(metric, ""), M, Mt)
            sub[col] = np.clip(pa, 0, None)
        summary["vc"] = vc
        log(f"varying-coefficient blend {vc}")
    # 2. residual correction (group + similarity kNN), weights frozen from the holdout fit
    # kNN configs (space, W, k, spatial) per metric come with their weights from the same holdout fit (PP0006)
    assert lamcfg["kshr"] == KSHR
    configs = {m: [tuple(c) for c in lamcfg["configs"][m]] for m in METRICS}
    summary["correction_configs"] = lamcfg["configs"]
    subn = tr["subzone_id"].str[4:].astype(int).values
    subnt = te["subzone_id"].str[4:].astype(int).values
    kt, ke = group_keys(tr, te, subn, subnt)
    Zs = spaces(tr, te)
    jobs, base = [], {}
    for metric, col in METRICS.items():
        fam = "elec" if metric == "elec" else "gas"
        yv = y[EL if metric == "elec" else EUI]
        pf, pa = blend_preds(P[fam][0], P[fam][1], yv, metric, A, At, vc.get(metric, ""), M, Mt)
        assert np.allclose(np.clip(pa, 0, None), sub[col].values, rtol=1e-9, atol=1e-9)
        r = (A if metric == "gas" else 1.0) * yv - pf
        base[metric] = pa
        assert len(lamcfg["lam"][metric]) == 3 + len(configs[metric])
        jobs.append((metric, kt, ke, r, pf, pa, A, At, Zs, len(tr), len(te), subn, subnt, lamcfg["lam"][metric],
                     configs[metric]))
    with ProcessPoolExecutor(3) as ex:
        for metric, corr in ex.map(_correction, jobs):
            col = METRICS[metric]
            sub[col] = np.clip(base[metric] + corr, 0, None)
            summary[f"correction_rms_over_sigma_{metric}"] = float(np.sqrt(np.mean(corr ** 2)) / SIGMAS[col])
            log(f"{metric} correction rms/sigma {summary[f'correction_rms_over_sigma_{metric}']:.5f}")
    # 2b. post-processing (D015), parameters frozen in resid_knn.json "postproc": S1 multiplicative shift of the rows
    # with very few meters (bld_dim_12 2-3 and 4, under-predicted by every member family); S3 truncated-normal shrink
    # towards the hard truncation of gas at the train maximum (gas and EUI); S4 clip to the train-implied bounds
    pp = lamcfg.get("postproc")
    if pp:
        from scipy.stats import norm
        B = dict(gas_max=tr[GAS].max(), eui_min=tr[EUI].min(), elecA_min=(tr[EL] / A).min(),
                 elecA_max=(tr[EL] / A).max())
        for metric, col in (("gas", GAS), ("eui", EUI), ("elec", EL)):
            p = sub[col].values.astype(float).copy()
            for key, g in (("M23", (Mt >= 2) & (Mt <= 3)), ("M4", Mt == 4)):
                p[g] *= 1 + pp["s1"][metric][key]
            if metric in pp["s3"]:
                c = pp["s3"][metric]
                s = p * np.sqrt(np.clip(c["a0"] + c["b0"] / Mt, 1e-8, None))
                b = ((B["gas_max"] if metric == "gas" else B["gas_max"] / At) - p) / s
                p = p + c["k"] * (-s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None))
            lo, hi = {"gas": (B["eui_min"] * At, np.full(len(At), B["gas_max"])),
                      "eui": (np.full(len(At), B["eui_min"]), B["gas_max"] / At),
                      "elec": (B["elecA_min"] * At, B["elecA_max"] * At)}[metric]
            sub[col] = np.clip(p, lo, hi)
        summary["postproc"] = pp
        log("post-processing S1 (small meters), S3 (gas truncation), S4 (train bounds) applied")
    # 3. test rows identical to a train row get its targets
    dup = duplicate_targets(tr, te)
    hit = ~np.isnan(dup[:, 0])
    for j, col in enumerate(TARGET_COLS):
        sub.loc[hit, col] = dup[hit, j]
    summary["duplicate_rows_copied"] = int(hit.sum())
    # 4. checks and output
    ss = load_sample()
    assert list(sub.columns) == list(ss.columns) and (sub["id"].values == ss["id"].values).all()
    assert np.isfinite(sub[TARGET_COLS].values).all()
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    sub.to_csv(out_csv, index=False)
    sha = hashlib.sha256(open(out_csv, "rb").read()).hexdigest()
    summary["sha256"] = sha
    json.dump(summary, open(f"{work}/summary.json", "w"), indent=1)
    log(f"wrote {out_csv} sha256 {sha}")
    return sha
