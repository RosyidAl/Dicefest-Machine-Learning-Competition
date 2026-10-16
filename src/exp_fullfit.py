"""Full refit vs fold averaging, measured on the holdout split of exp_hold.py: a member is trained once on all
of P (GBMs: 1.1 x the mean early-stopping iteration of its 10-fold hold run; MLPs: the same epoch count) for
several seeds and predicts H. Per-seed H predictions go to exp/hold/full/<name>_s<seed>.npy and are scored
against the 10-fold-averaged H predictions in exp/hold/pred.

usage: python src/exp_fullfit.py name1,name2,... [seeds, default 42,43,44]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import TARGET_COLS, nrmse  # noqa: E402
from exp_hold import CPU, GPU, H, TE, W, d, nk, s  # noqa: E402
from nn import fit_fold, preprocess, set_seed  # noqa: E402
from train_lib import is_gas_kind, is_log_kind, target_spec  # noqa: E402

GAS, ELE, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
OUT = "exp/hold/full"


def hold_iters(name):
    for f in ("exp/hold_gpu.log", "exp/hold_xgbrev.log", "exp/hold_cpu.log"):
        for line in open(f) if os.path.exists(f) else []:
            if line.startswith(f"[{name}] "):
                return [int(v) for v in re.search(r"iters=\[([^\]]*)\]", line).group(1).split(",")]
    raise KeyError(name)


def fit_fixed(backend, params, Xa, ya, wa, Xt, n_iter, seed):
    if backend == "xgb":
        import xgboost as xgb
        p = dict(params, device="cuda", tree_method="hist", seed=seed)
        da = xgb.QuantileDMatrix(Xa, ya, weight=wa, max_bin=p.get("max_bin", 256))
        return xgb.train(p, da, n_iter).inplace_predict(Xt)
    if backend == "lgb":
        import lightgbm as lgb
        p = dict(objective="regression", seed=seed, deterministic=True, force_row_wise=True, num_threads=16,
                 verbose=-1)
        p.update(params)
        return lgb.train(p, lgb.Dataset(Xa, ya, weight=wa), n_iter).predict(Xt)
    from catboost import CatBoostRegressor, Pool
    m = CatBoostRegressor(loss_function="RMSE", task_type="GPU", random_seed=seed, verbose=0, iterations=n_iter,
                          gpu_ram_part=0.45, **params)
    m.fit(Pool(Xa, ya, weight=wa))
    return m.predict(Xt)


def nn_inputs(Fa, Ft):
    """Same encoding as nn.run_nn/te_block, with all of P as the fitting part."""
    Z, Zt = preprocess(s.X, s.Xt)
    Fa, Ft = Fa.copy(), Ft.copy()
    cnt = [c for c in Fa.columns if c.endswith("_cnt")]
    for F in (Fa, Ft):
        F[cnt] = np.log1p(F[cnt])
    mu, sd = Fa.mean(), Fa.std().replace(0, 1)
    T = [np.hstack([((F - mu) / sd).clip(-6, 6).fillna(0).values, F.isna().astype("float32").values])
         .astype("float32") for F in (Fa, Ft)]
    return np.hstack([Z, T[0]]), np.hstack([Zt, T[1]])


def score(kind, p):
    if is_gas_kind(kind):
        return f"eui {nrmse(d.y[EUI][H], p, EUI):.5f} gas {nrmse(d.y[GAS][H], p * d.A[H], GAS):.5f}"
    return f"elec {nrmse(d.y[ELE][H], p, ELE):.5f}"


def main():
    names = sys.argv[1].split(",")
    seeds = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 43, 44]
    os.makedirs(OUT, exist_ok=True)
    spec = {r[0]: r for r in GPU + CPU}
    allr = np.arange(len(s.X))
    Fa = s.te_features(allr, "oof", TE["windows"], TE["cells"])
    Ft = s.te_features(allr, s.te_apply(), TE["windows"], TE["cells"])
    Xa = pd.concat([s.X.reset_index(drop=True), Fa], axis=1)
    Xt = pd.concat([s.Xt.reset_index(drop=True), Ft], axis=1)
    Za = Zt = None
    for name in names:
        _, backend, kind, params, wk = spec[name]
        y, w, _ = target_spec(s, kind)
        mult_t = target_spec(s, kind, test=True)[2]
        if W[wk] is not None:
            w = W[wk] / W[wk].mean() if w is None else w * W[wk] / np.mean(w * W[wk])
        preds = []
        for seed in seeds:
            f = f"{OUT}/{name}_s{seed}.npy"
            if not os.path.exists(f):
                if backend == "nn":
                    if Za is None:
                        Za, Zt = nn_inputs(Fa, Ft)
                    set_seed(seed)
                    kw = {k: v for k, v in nk.items() if k != "seeds"}
                    p = fit_fold(Za, y, w, Zt, np.zeros(len(Zt)), None, Zt, epochs=params, fixed_ep=params - 1,
                                 **kw)[1]
                else:
                    n_iter = int(round(1.1 * (np.mean(hold_iters(name)) + (backend != "lgb"))))
                    p = fit_fixed(backend, params, Xa, y, w, Xt, n_iter, seed)
                np.save(f, (np.exp(p) if is_log_kind(kind) else p) * mult_t)
            preds.append(np.load(f))
        fold = np.load(f"exp/hold/pred/{name}.npy")
        full = np.mean(preds, axis=0)
        print(f"[{name}] fold-avg {score(kind, fold)} | full x1 {score(kind, preds[0])} | "
              f"full x{len(preds)} {score(kind, full)} | mix {score(kind, (fold + full) / 2)}", flush=True)


if __name__ == "__main__":
    main()
