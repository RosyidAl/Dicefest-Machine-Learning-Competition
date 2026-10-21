"""Re-create every member's OOF and test predictions from the saved fold models (no training).

Mirrors train_lib.run / nn.run_nn / nn.run_nn_multi step by step: the same features (target encodings of each fold's
fitting part, aggregates), the same prediction calls (XGBoost inplace_predict on CUDA up to the best iteration,
LightGBM saved at its best iteration, MLP eval-mode prediction in the same chunks), the same back-transforms.
Used by inference_from_artifacts.ipynb; check_member() compares the result with the arrays of a training run.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from common import SEED
from pipeline import load_config


def _gbm_predict(backend, path, Xb, Xt):
    if backend == "xgb":
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(path + ".ubj")
        m.set_param({"device": "cuda"})
        rng = (0, m.best_iteration + 1)
        return m.inplace_predict(Xb, iteration_range=rng), m.inplace_predict(Xt, iteration_range=rng)
    import lightgbm as lgb
    m = lgb.Booster(model_file=path + ".txt")
    return m.predict(Xb), m.predict(Xt)


def predict_gbm(d, name, spec, model_dir, extra=None, extra_t=None):
    from train_lib import is_log_kind, target_spec
    kind = spec["kind"]
    _, _, mult = target_spec(d, kind)
    mult_t = target_spec(d, kind, test=True)[2]
    use_te, windows, cells = spec.get("use_te", False), tuple(spec.get("windows", ())), spec.get("cells", False)
    oof, pt = np.zeros(len(d.X)), np.zeros(len(d.Xt))
    for k, (a, b) in enumerate(d.folds):
        Xb, Xt = d.X.iloc[b], d.Xt
        if extra is not None:
            Xb = pd.concat([Xb.reset_index(drop=True), extra.iloc[b].reset_index(drop=True)], axis=1)
            Xt = pd.concat([Xt.reset_index(drop=True), extra_t.reset_index(drop=True)], axis=1)
        if use_te:
            Xb = pd.concat([Xb.reset_index(drop=True), d.te_features(a, d.te_apply(b), windows, cells)], axis=1)
            Xt = pd.concat([Xt.reset_index(drop=True), d.te_features(a, d.te_apply(), windows, cells)], axis=1)
        if spec.get("knn"):
            Xb = pd.concat([Xb.reset_index(drop=True), d.knn_features(a, ("train", b))], axis=1)
            Xt = pd.concat([Xt.reset_index(drop=True), d.knn_features(a, "test")], axis=1)
        pb, ptk = _gbm_predict(spec["backend"], f"{model_dir}/fold{k}", Xb, Xt)
        if is_log_kind(kind):
            pb, ptk = np.exp(pb), np.exp(ptk)
        oof[b] = pb * (mult if np.isscalar(mult) else mult[b])
        pt += ptk * mult_t / len(d.folds)
    return oof, pt


def _load_mlp(path):
    from nn import MLP
    ck = torch.load(path + ".pt", map_location="cuda", weights_only=False)
    ar = ck["arch"]
    model = MLP(ar["d_in"], ar["hidden"], ar["drop"], ar["emb"], ck["cont"], ar["ens"], n_out=ar["n_out"]).to("cuda")
    model.load_state_dict(ck["state_dict"])
    return model, ar, ck["mu"], ck["sd"]


def _nn_inputs(d, extra, extra_t):
    from nn import preprocess
    if extra is None:
        return preprocess(d.X, d.Xt)
    return preprocess(pd.concat([d.X.reset_index(drop=True), extra], axis=1),
                      pd.concat([d.Xt.reset_index(drop=True), extra_t], axis=1))


def predict_nn(d, name, spec, model_dir, extra=None, extra_t=None, tasks=None):
    """single-task (spec['kind']) or multi-task (tasks = [(suffix, kind, wmul code)]) MLP"""
    from nn import predict_model, set_seed, te_block
    from train_lib import is_log_kind, target_spec
    set_seed()
    Z, Zt = _nn_inputs(d, extra, extra_t)
    kinds = [spec["kind"]] if tasks is None else [t[1] for t in tasks]
    mults = [target_spec(d, k)[2] for k in kinds]
    mults_t = [target_spec(d, k, test=True)[2] for k in kinds]
    seeds = spec["seeds"]
    oof, pt = np.zeros((len(Z), len(kinds))), np.zeros((len(Zt), len(kinds)))
    for k, (a, b) in enumerate(d.folds):
        Zb, Ztk = Z[b], Zt
        if spec.get("use_te"):
            _, Tb, Tt = te_block(d, a, b, tuple(spec["windows"]), spec["cells"], spec.get("knn", False))
            Zb, Ztk = np.hstack([Zb, Tb]), np.hstack([Zt, Tt])
        Xb, Xt = torch.tensor(Zb, device="cuda"), torch.tensor(Ztk, device="cuda")
        for s in seeds:
            model, ar, mu, sd = _load_mlp(f"{model_dir}/fold{k}_seed{s}")
            pb, ptk = predict_model(model, Xb, ar["emb"], ar["ens"], mu, sd), predict_model(model, Xt, ar["emb"],
                                                                                           ar["ens"], mu, sd)
            pb, ptk = pb.reshape(len(b), -1), ptk.reshape(len(Zt), -1)
            for j, kind in enumerate(kinds):
                qb, qt = (np.exp(pb[:, j]), np.exp(ptk[:, j])) if is_log_kind(kind) else (pb[:, j], ptk[:, j])
                m = mults[j]
                oof[b, j] += qb * (m if np.isscalar(m) else m[b]) / len(seeds)
                pt[:, j] += qt * mults_t[j] / len(d.folds) / len(seeds)
    return oof, pt


def predict_nca(d, name, spec, model_dir, extra=None, extra_t=None):
    """NCA members (nca.run_nca / nca.fit_fold_nca): rebuild each fold's inputs, load the fold model, embed the fold's
    training rows and predict validation and test rows with the same chunking as the training run"""
    import pandas as pd
    import torch
    from nca import Embed, _dist
    from nn import preprocess, te_block
    from train_lib import is_log_kind, target_spec
    p = dict(spec["params"])
    hidden, d_out, drop = tuple(p.get("hidden", (512,))), p.get("d_out", 128), p.get("drop", 0.1)
    emb, chunk = p.get("emb"), p.get("chunk", 128)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if extra is None:
        Z, Zt = preprocess(d.X, d.Xt)
    else:
        Z, Zt = preprocess(pd.concat([d.X.reset_index(drop=True), extra], axis=1),
                           pd.concat([d.Xt.reset_index(drop=True), extra_t], axis=1))
    kind = spec["kind"]
    y, _, mult = target_spec(d, kind)
    mult_t = target_spec(d, kind, test=True)[2]
    seeds = tuple(spec["seeds"])
    oof, pt = np.zeros(len(Z)), np.zeros(len(Zt))
    for k, (a, b) in enumerate(d.folds):
        Za, Zb, Ztk = Z[a], Z[b], Zt
        if spec.get("use_te", True):
            Ta, Tb, Tt = te_block(d, a, b, tuple(spec.get("windows", (2, 10))), spec.get("cells", True),
                                  spec.get("knn", True))
            Za, Zb, Ztk = np.hstack([Za, Ta]), np.hstack([Zb, Tb]), np.hstack([Zt, Tt])
        ya = y[a]
        mu, sd = ya.mean(), ya.std()
        for s in seeds:
            cont = None if emb is None else np.array([len(np.unique(Za[:, j])) > 2 for j in range(Za.shape[1])])
            model = Embed(Za.shape[1], hidden, d_out, drop, emb, cont).to(dev)
            model.load_state_dict(torch.load(f"{model_dir}/fold{k}_seed{s}.pt", map_location=dev))
            model.eval()
            with torch.no_grad():
                Xa = torch.tensor(Za, device=dev)
                Ya = torch.tensor((ya - mu) / sd, dtype=torch.float32, device=dev)
                Ea = torch.cat([model(Xa[i:i + 65536]) for i in range(0, len(Za), 65536)])

                def predict(Zx):
                    X = torch.tensor(Zx, device=dev)
                    out = []
                    for i in range(0, len(X), chunk):
                        logit = -_dist(model(X[i:i + chunk]), Ea)
                        out.append((torch.softmax(logit, 1) * Ya[None]).sum(1))
                    return torch.cat(out).cpu().numpy() * sd + mu

                pb, ptk = predict(Zb), predict(Ztk)
            if is_log_kind(kind):
                pb, ptk = np.exp(pb), np.exp(ptk)
            oof[b] += pb * (mult if np.isscalar(mult) else mult[b]) / len(seeds)
            pt += ptk * mult_t / len(d.folds) / len(seeds)
    return oof, pt


def predict_all(work_models, out_dir, only=None):
    """writes <out_dir>/oof|pred/<member>.npy for every member (or `only`) from <work_models>/<member>/"""
    from pipeline import make_data
    cfg = load_config()
    os.makedirs(f"{out_dir}/oof", exist_ok=True)
    os.makedirs(f"{out_dir}/pred", exist_ok=True)
    d = make_data(out_dir)
    ex, ext = d.agg_features()
    done = set()
    for name, spec in cfg["members"].items():
        if only and name not in only:
            continue
        if spec["backend"] == "nn_multi":
            g = spec["group"]
            if g in done:
                continue
            done.add(g)
            m = cfg["multi"][g]
            d.folds = list(KFold(m["n_folds"], shuffle=True, random_state=SEED).split(d.X))
            d.use_x = bool(m.get("knn_x", False))
            agg = (ex, ext) if m["agg"] else (None, None)
            oof, pt = predict_nn(d, g, m, f"{work_models}/{g}", *agg, tasks=m["tasks"])
            for j, t in enumerate(m["tasks"]):
                np.save(f"{out_dir}/oof/{g}_{t[0]}.npy", oof[:, j])
                np.save(f"{out_dir}/pred/{g}_{t[0]}.npy", pt[:, j])
            continue
        d.folds = list(KFold(spec["n_folds"], shuffle=True, random_state=SEED).split(d.X))
        d.use_x = bool(spec.get("knn_x", False))
        agg = (ex, ext) if spec["agg"] else (None, None)
        if spec["backend"] == "nn":
            oof, pt = predict_nn(d, name, spec, f"{work_models}/{name}", *agg)
            oof, pt = oof[:, 0], pt[:, 0]
        elif spec["backend"] == "nca":
            oof, pt = predict_nca(d, name, spec, f"{work_models}/{name}", *agg)
        else:
            oof, pt = predict_gbm(d, name, spec, f"{work_models}/{name}", *agg)
        np.save(f"{out_dir}/oof/{name}.npy", oof)
        np.save(f"{out_dir}/pred/{name}.npy", pt)
        print(f"[{name}] predicted from saved models", flush=True)


def check_member(out_dir, ref_dir, name):
    res = {}
    for part in ("oof", "pred"):
        a, b = np.load(f"{out_dir}/{part}/{name}.npy"), np.load(f"{ref_dir}/{part}/{name}.npy")
        res[part] = "identical" if np.array_equal(a, b) else f"max abs diff {np.abs(a - b).max():.3g}"
    return res
