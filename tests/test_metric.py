import numpy as np
import pandas as pd

from metric import OFFICIAL_SIGMAS, TARGET_COLS, compute_column_wise_nrmse, nrmse_per_target


def test_matches_manual_numpy():
    rng = np.random.default_rng(0)
    yt = pd.DataFrame(rng.normal(size=(1000, 5)) * 100, columns=TARGET_COLS)
    yp = yt + rng.normal(size=yt.shape) * 10
    manual = np.mean([np.sqrt(np.mean((yt[c] - yp[c]) ** 2)) / OFFICIAL_SIGMAS[c] for c in TARGET_COLS])
    assert np.isclose(compute_column_wise_nrmse(yt, yp), manual, rtol=1e-12)
    per = nrmse_per_target(yt, yp)
    assert np.isclose(np.mean(list(per.values())), manual, rtol=1e-12)


def test_mean_prediction_is_about_one():
    tr = pd.read_parquet("data/cache/train.parquet", columns=TARGET_COLS)
    pred = pd.DataFrame({c: np.full(len(tr), tr[c].mean()) for c in TARGET_COLS})
    per = nrmse_per_target(tr, pred)
    assert all(0.9 < v < 1.1 for v in per.values()), per
