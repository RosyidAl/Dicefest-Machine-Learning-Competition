"""Phase 8: blending methods compared under nested CV (weights learned on 4/5 of the OOF rows, scored on the
remaining 1/5, rotated over 5 folds) on the v15nc members. Metrics: gas (EUI members x floor area, gas-scale loss),
EUI and elec. Output: exp/analysis/ensemble_methods.json + printed table."""
import json
import re
import sys

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS, load  # noqa: E402

log = open("exp/prune_v15nc.log").read()
gas = re.search(r"^gas runs: (\S+)", log, re.M).group(1).split(",")
el = re.search(r"^elec runs: (\S+)", log, re.M).group(1).split(",")
tr, _ = load()
A = tr["bld_dim_06"].values
Pg = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in gas])
Pe = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in el])
CASES = {"gas": (Pg, tr[TARGET_COLS[2]].values, A, TARGET_COLS[0]),
         "eui": (Pg, tr[TARGET_COLS[2]].values, np.ones(len(A)), TARGET_COLS[2]),
         "elec": (Pe, tr[TARGET_COLS[1]].values, np.ones(len(A)), TARGET_COLS[1])}


def rmse(p, y, s):
    return np.sqrt(np.mean((s * (p - y)) ** 2))


def m_single(P, y, s):
    j = int(np.argmin([rmse(P[:, k], y, s) for k in range(P.shape[1])]))
    w = np.zeros(P.shape[1]); w[j] = 1
    return w


def m_topn(P, y, s):
    order = np.argsort([rmse(P[:, k], y, s) for k in range(P.shape[1])])
    best = None
    for n in range(2, 11):
        w = np.zeros(P.shape[1]); w[order[:n]] = 1 / n
        sc = rmse(P @ w, y, s)
        if best is None or sc < best[0]:
            best = (sc, w)
    return best[1]


def m_nnls(P, y, s):
    return nnls(P * s[:, None], y * s)[0]


def m_hill(P, y, s, iters=100, bags=20, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros(P.shape[1])
    for _ in range(bags):
        sub = np.sort(rng.choice(P.shape[1], max(3, P.shape[1] // 2), replace=False))
        sc = [rmse(P[:, k], y, s) for k in sub]
        cnt = np.zeros(P.shape[1])
        for k in np.array(sub)[np.argsort(sc)[:3]]:
            cnt[k] += 1
        cur = P @ cnt / cnt.sum()
        for _ in range(iters):
            n = cnt.sum()
            cand = [rmse((cur * n + P[:, k]) / (n + 1), y, s) for k in sub]
            k = sub[int(np.argmin(cand))]
            cnt[k] += 1
            cur = (cur * n + P[:, k]) / (n + 1)
        W += cnt / cnt.sum() / bags
    return W


class RidgePos:
    def __call__(self, P, y, s):
        self.m = Ridge(alpha=1e-6 * len(y), positive=True, fit_intercept=True).fit(P, y, sample_weight=s ** 2)
        return self

    def predict(self, P):
        return self.m.predict(P)


class MLP2:
    def __call__(self, P, y, s):
        self.mu, self.sd = P.mean(0), P.std(0)
        self.ym, self.ys = y.mean(), y.std()
        self.m = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=60, early_stopping=True, random_state=0, batch_size=4096)
        self.m.fit((P - self.mu) / self.sd, (y - self.ym) / self.ys)
        return self

    def predict(self, P):
        return self.m.predict((P - self.mu) / self.sd) * self.ys + self.ym


METHODS = {"EN1 single best": m_single, "EN2 top-N avg": m_topn, "EN3 NNLS (dipakai)": m_nnls,
           "EN4 hill climbing x20 bag": m_hill, "EN5 Ridge positive + intercept": RidgePos, "EN7 MLP level-2": MLP2}
res = {}
for case, (P, y, s, col) in CASES.items():
    res[case] = {}
    for name, meth in METHODS.items():
        pred = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=0).split(P):
            if isinstance(meth, type):
                mdl = meth()(P[a], y[a], s[a])
                pred[b] = mdl.predict(P[b])
            else:
                pred[b] = P[b] @ meth(P[a], y[a], s[a])
        res[case][name] = float(rmse(pred, y, s) / SIGMAS[col])
        print(f"{case:4s} {name:32s} {res[case][name]:.5f}", flush=True)
tot = {n: sum(res[c][n] for c in res) / 5 for n in METHODS}
for n, v in tot.items():
    print(f"total contribution (gas+eui+elec)/5  {n:32s} {v:.5f}  vs NNLS {v - tot['EN3 NNLS (dipakai)']:+.5f}")
json.dump({"per_metric": res, "total_contribution": tot}, open("exp/analysis/ensemble_methods.json", "w"), indent=1)
