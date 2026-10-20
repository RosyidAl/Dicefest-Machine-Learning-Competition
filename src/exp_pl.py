"""PL0001: pseudo-labelling (teacher = VC blend, student = one member config trained on true + pseudo targets).

chain : teacher = varying-coefficient blend A2xM2xp2 of all hold members (fitted on P-OOF, applied to H, no residual
        correction). Student = a member config trained 10-fold on the union of P (true targets) and H (teacher
        predictions as targets). The student's H predictions are its out-of-fold predictions of the H rows, so each H
        row is predicted by models that never saw its own pseudo target, and no true H target is used anywhere.
        -> exp/pl/chain/{oof,pred}/<name>.npy (union order: P rows, then H rows)
deploy: same with train rows (true targets) + test rows (deployed VC blend of the v19 members, no correction).
        -> exp/pl/deploy/{oof,pred}/<name>.npy (union order: train rows, then test rows)
Target-dependent features (subzone / zone / cell / window encodings, FB07 + FB07b kNN-target block) are computed
inside each fold from the fold's fitting rows, so pseudo targets of other held-out rows enter them like any target.
Pseudo rows get training weight w (default 1.0) times the member's own weight.

usage: python src/exp_pl.py chain|deploy <name> [w] [smoke]
"""
import copy
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, TARGET_COLS, holdout_split, load  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
SPEC = "A2xM2xp2"
lp = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
x8 = dict(learning_rate=0.03, max_depth=10, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0,
          eval_metric="rmse")
x4 = dict(x8, learning_rate=0.05, max_depth=8)
XP = dict(x4, objective="count:poisson", eval_metric="poisson-nloglik")
PLR = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44), emb=dict(k=24, sigma=0.05, d=24),
           epochs=12, fixed_ep=11)
JOBS = {  # same configs as the src/exp_v18_knn2.py members of the same base name
    "lp2_gpm": ("lgb", "gpm", dict(lp, objective="poisson"), "iM"),
    "lp2_epm": ("lgb", "epm", dict(lp, objective="poisson"), "iM"),
    "xp2_gpm": ("xgb", "gpm", dict(XP, min_child_weight=1e6), "iM"),
    "xp2_epm": ("xgb", "epm", dict(XP, min_child_weight=2.8e5), "iM"),
    "pk2_gpgea": ("nn", "gas_per:bld_dim_03", PLR, "iA"),
    "pk2_lepm": ("nn", "lepm", dict(PLR, emb=dict(k=24, sigma=0.02, d=48), epochs=10, fixed_ep=9), "M"),
}


def hold_names():
    from prune import ELEC
    from resid_corr import X8
    is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
    mem = sorted(f[:-4] for f in os.listdir("exp/hold/pred") if os.path.getsize(f"exp/hold/pred/{f}") > 0)
    return [X8.get(m, m) for m in mem if not is_el(m)] + ["p1_gpm"], [m for m in mem if is_el(m)] + ["p1_lepm"]


def teacher(mode, tr, te):
    """teacher predictions on the pseudo rows (H for chain, test for deploy): gas, eui, elec (metric scale); cached"""
    f = f"exp/pl/teacher_{mode}.npz"
    if os.path.exists(f):
        z = np.load(f)
        return {k: z[k] for k in ("gas", "eui", "elec")}
    from blend import blend_vc
    A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
    if mode == "chain":
        P, H = holdout_split(len(tr))
        gas, el = hold_names()
        path = lambda n, part: next(p for p in (f"exp/hold/{part}/{n}.npy", f"exp/hold/tune/{part}/{n}.npy")  # noqa
                                    if os.path.exists(p))
        cols = {fam: (np.column_stack([np.load(path(n, "oof")) for n in names]),
                      np.column_stack([np.load(path(n, "pred")) for n in names])) for fam, names in (("gas", gas), ("elec", el))}
        fit, Af, Aa, Mf, Ma = P, A[P], A[H], M[P], M[H]
    else:
        import yaml
        cfg = yaml.safe_load(open("final/configs/members.yaml"))["blend"]
        cols = {fam: (np.column_stack([np.load(f"exp/oof/{n}.npy") for n in cfg[fam]]),
                      np.column_stack([np.load(f"exp/pred/{n}.npy") for n in cfg[fam]])) for fam in ("gas", "elec")}
        fit, Af, Aa = np.arange(len(tr)), A, te["bld_dim_06"].values
        Mf, Ma = M, te["bld_dim_12"].values.astype(float)
    out = {}
    for metric in ("gas", "eui", "elec"):
        Pf, Pa = cols["elec" if metric == "elec" else "gas"]
        yv = tr[EL if metric == "elec" else EUI].values[fit]
        _, pa, _ = blend_vc(Pf, Pa, yv, SPEC, dict(A=Af, M=Mf), dict(A=Aa, M=Ma), Af if metric == "gas" else None)
        out[metric] = np.clip(pa * (Aa if metric == "gas" else 1.0), 1e-6, None)
    os.makedirs("exp/pl", exist_ok=True)
    np.savez(f, **out)
    return out


def main(mode, name, w_pl=1.0, smoke=False):
    from fb07b import DataX
    from resid_knn import spaces
    from train_lib import run
    tr, te = load()
    d = DataX(n_folds=10)
    S = spaces(tr, te)
    n = len(tr)
    d.sp = {k: (v[:n], v[n:]) for k, v in S.items()}
    ex, ext = d.agg_features()
    T = teacher(mode, tr, te)
    import pandas as pd
    if mode == "chain":
        P, H = holdout_split(n)
        perm = np.random.default_rng(7).permutation(n)
        assert (P == np.sort(perm[: int(0.7 * n)])).all()  # same split as the hold members
        # union rows: P (true) then H (pseudo); every array below is indexed from the train arrays
        X = pd.concat([d.X.iloc[P], d.X.iloc[H]], ignore_index=True)
        A, M = np.r_[d.A[P], d.A[H]], np.r_[d.M[P], d.M[H]]
        sub, zone, subn = (np.r_[a[P], a[H]] for a in (d.sub, d.zone, d.subn))
        cells = [np.r_[c[P], c[H]] for c in d.cells]
        sp = {k: np.r_[v[0][P], v[0][H]] for k, v in d.sp.items()}
        exu = pd.concat([ex.iloc[P], ex.iloc[H]], ignore_index=True)
        ytrue = {c: d.y[c][P] for c in TARGET_COLS}
        n_true = len(P)
    else:
        X = pd.concat([d.X, d.Xt], ignore_index=True)
        A, M = np.r_[d.A, d.At], np.r_[d.M, d.Mt]
        sub, zone, subn = np.r_[d.sub, d.subt], np.r_[d.zone, d.zonet], np.r_[d.subn, d.subnt]
        cells = [np.r_[c, ct] for c, ct in zip(d.cells, d.cellst)]
        sp = {k: np.r_[v[0], v[1]] for k, v in d.sp.items()}
        exu = pd.concat([ex, ext], ignore_index=True)
        ytrue = d.y
        n_true = n
    n_u = len(X)
    pseudo = {GAS: T["gas"], EL: T["elec"], EUI: T["eui"]}
    y = {c: np.r_[ytrue[c], pseudo[c] if c in pseudo else np.zeros(n_u - n_true)] for c in TARGET_COLS}
    keep = np.arange(n_u)
    if smoke:  # code check: small random subset, 2 folds, few trees
        keep = np.sort(np.random.default_rng(0).choice(n_u, 40000, replace=False))
    s = copy.copy(d)
    s.X = X.iloc[keep].reset_index(drop=True)
    s.Xt = s.X.iloc[:2000].reset_index(drop=True)  # unused dummy apply rows (the student output is its OOF)
    dummy = keep[:2000]
    s.A, s.M, s.At, s.Mt = A[keep], M[keep], A[dummy], M[dummy]
    s.y = {c: v[keep] for c, v in y.items()}
    s.sub, s.zone, s.subn = sub[keep], zone[keep], subn[keep]
    s.subt, s.zonet, s.subnt = sub[dummy], zone[dummy], subn[dummy]
    s.cells, s.cellst = [c[keep] for c in cells], [c[dummy] for c in cells]
    s.sp = {k: (v[keep], v[dummy]) for k, v in sp.items()}
    yg, ye, yl = s.y[GAS], s.y[EUI], s.y[EL]
    s.V = np.c_[ye, np.log(ye), yl / s.M, np.log(yl / s.M), yl / s.A, yg / s.M, np.log(yg)]
    s.W = np.c_[np.log(s.A), np.log(s.M)]
    s._kz = None
    s.folds = list(KFold(10, shuffle=True, random_state=SEED).split(s.X))
    if smoke:
        s.folds = s.folds[:2]
    tag = f"{mode}{'_smoke' if smoke else ''}"
    s.knn_cache_dir = f"cache/pl_{tag}/knn"
    s.knn_x_cache_dir = f"cache/pl_{tag}/knn_x"
    is_pseudo = keep >= n_true
    backend, kind, params, wk = JOBS[name]
    params = dict(params)
    if smoke:
        params["n_estimators"] = 50
        params.update(epochs=2, fixed_ep=1) if backend == "nn" else None
    W = {"A": s.A, "iA": 1.0 / s.A, "M": s.M.astype(float), "iM": 1.0 / s.M, None: np.ones(len(s.A))}[wk]
    W = W * np.where(is_pseudo, w_pl, 1.0)
    out = f"exp/pl/{tag}"
    for part in ("oof", "pred"):  # nn.run_nn does not create its output directories
        os.makedirs(f"{out}/{part}", exist_ok=True)
    run_name = f"{name}_w{w_pl:g}"
    agg = dict(extra=exu.iloc[keep].reset_index(drop=True), extra_t=exu.iloc[dummy].reset_index(drop=True))
    if backend == "nn":
        from nn import run_nn
        run_nn(s, run_name, kind, out_dir=out, wmul=W, use_te=True, windows=(2, 10), cells=True, knn=True, **agg,
               **params)
    else:
        run(s, run_name, backend, kind, params, wmul=W, use_te=True, windows=(2, 10), cells=True, out_dir=out,
            knn=True, **agg)
    np.save(f"{out}/keep_{run_name}.npy", keep)
    if mode == "deploy" and not smoke:  # student test predictions = out-of-fold predictions of the test rows
        np.save(f"{out}/test_{run_name}.npy", np.load(f"{out}/oof/{run_name}.npy")[n_true:])
    print("done", out, run_name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0, "smoke" in sys.argv[4:])
