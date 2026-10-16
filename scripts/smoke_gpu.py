"""Phase 0 smoke test: tiny fits on XGBoost CUDA, CatBoost GPU, PyTorch CUDA and LightGBM CPU (prints versions)."""
import time

import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(20000, 20)).astype("float32")
y = X[:, 0] * 2 + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=len(X))
out = []


def run(name, fn):
    t = time.time()
    try:
        msg = fn()
        out.append(f"| {name} | OK | {msg} | {time.time() - t:.1f}s |")
    except Exception as e:  # noqa: BLE001
        out.append(f"| {name} | FAIL | {type(e).__name__}: {str(e)[:120]} | |")


def xgb_cuda():
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=50, device="cuda", tree_method="hist", random_state=42).fit(X, y)
    return f"xgboost {xgb.__version__}, rmse {np.sqrt(np.mean((m.predict(X) - y) ** 2)):.4f}"


def cat_gpu():
    import catboost as cb
    m = cb.CatBoostRegressor(iterations=50, task_type="GPU", random_seed=42, verbose=0).fit(X, y)
    return f"catboost {cb.__version__}, rmse {np.sqrt(np.mean((m.predict(X) - y) ** 2)):.4f}"


def torch_cuda():
    import torch
    a = torch.randn(2048, 2048, device="cuda")
    return f"torch {torch.__version__}, cuda {torch.version.cuda}, {torch.cuda.get_device_name(0)}, sum {(a @ a).sum().item():.1f}"


def lgb_cpu():
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=50, num_threads=4, deterministic=True, force_row_wise=True, seed=42,
                          verbose=-1).fit(X, y)
    return f"lightgbm {lgb.__version__}, rmse {np.sqrt(np.mean((m.predict(X) - y) ** 2)):.4f}"


def faiss_gpu():
    import faiss
    res = faiss.StandardGpuResources()
    idx = faiss.GpuIndexFlatL2(res, X.shape[1])
    idx.add(X)
    return f"faiss {faiss.__version__}, gpus {faiss.get_num_gpus()}, knn ok {idx.search(X[:5], 3)[1].shape}"


for n, f in [("XGBoost CUDA", xgb_cuda), ("CatBoost GPU", cat_gpu), ("PyTorch CUDA", torch_cuda),
             ("LightGBM CPU", lgb_cpu), ("FAISS GPU", faiss_gpu)]:
    run(n, f)
print("| Komponen | Status | Detail | Waktu |\n|---|---|---|---|")
print("\n".join(out))
