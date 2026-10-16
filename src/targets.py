"""Target transforms: forward(y, ctx) -> (y_t, weight), inverse(pred_t, ctx) -> pred on the original scale.

ctx is a dict of per-row arrays for the rows being transformed (exposure columns, e.g. ctx["bld_dim_06"]).
Fitted transforms (yeo-johnson) learn their parameter in fit() on the training-fold rows only.
"""
import numpy as np
from scipy import stats


class Transform:
    def __init__(self, type="identity", exposure_col=None, weight_power=0, eps=0.5, weight_col=None,
                 weight_col_power=0, **kw):
        """weight_col / weight_col_power: extra sample-weight factor ctx[weight_col] ** power (e.g. 1 / floor area)"""
        self.type, self.col, self.p, self.eps, self.kw = type, exposure_col, weight_power, eps, kw
        self.wcol, self.wp = weight_col, weight_col_power
        self.lmbda = None

    def fit(self, y, ctx):
        if self.type == "yeojohnson":
            self.lmbda = float(stats.yeojohnson(y)[1])
        return self

    def forward(self, y, ctx):
        t, w = self.type, None
        if t == "identity":
            yt = y
        elif t == "log1p":
            yt = np.log1p(y)
        elif t == "sqrt":
            yt = np.sqrt(np.clip(y, 0, None))
        elif t == "yeojohnson":
            yt = stats.yeojohnson(y, lmbda=self.lmbda)
        elif t == "ratio":
            E = ctx[self.col].astype(float)
            yt = y / E
            w = E ** self.p if self.p else None
        elif t == "logratio":
            yt = np.log(y / ctx[self.col].astype(float))
        elif t == "proportion":
            yt = np.clip(y / 100.0, 0, 1)
        elif t == "logit":
            q = (y + self.eps) / (100 + 2 * self.eps)
            yt = np.log(q / (1 - q))
        else:
            raise ValueError(t)
        if self.wcol:
            extra = ctx[self.wcol].astype(float) ** self.wp
            w = extra if w is None else w * extra
        if w is not None:
            w = w / w.mean()
        return yt, w

    def inverse(self, pt, ctx):
        t = self.type
        if t == "identity":
            return pt
        if t == "log1p":
            return np.expm1(pt)
        if t == "sqrt":
            return np.clip(pt, 0, None) ** 2
        if t == "yeojohnson":
            lm = self.lmbda
            out = np.empty_like(pt, dtype=float)
            pos = pt >= 0
            if abs(lm) > 1e-12:
                out[pos] = np.power(np.clip(pt[pos] * lm + 1, 1e-12, None), 1 / lm) - 1
            else:
                out[pos] = np.expm1(pt[pos])
            if abs(lm - 2) > 1e-12:
                out[~pos] = 1 - np.power(np.clip(-(2 - lm) * pt[~pos] + 1, 1e-12, None), 1 / (2 - lm))
            else:
                out[~pos] = -np.expm1(-pt[~pos])
            return out
        if t == "ratio":
            return pt * ctx[self.col].astype(float)
        if t == "logratio":
            return np.exp(pt) * ctx[self.col].astype(float)
        if t == "proportion":
            return 100.0 * np.clip(pt, 0, 1)
        if t == "logit":
            return (100 + 2 * self.eps) / (1 + np.exp(-pt)) - self.eps
        raise ValueError(t)
