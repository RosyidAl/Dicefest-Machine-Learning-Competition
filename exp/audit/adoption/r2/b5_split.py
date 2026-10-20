"""Adoption lens (round 2), check b5: are the unseen test subzones exchangeable with labelled subzones of equal size?

The adoption CV proxy (labelled subzones reweighted to the unseen nrow mix) is unbiased only if a subzone ends up
entirely in test by chance (random row split) and not by a feature-dependent selection.
 1. counts: expected number of fully-test subzones per total row count n under a random row split
    (hypergeometric, exact) vs observed
 2. adversarial validation per stratum nrow = 1 and nrow = 2: LightGBM classifier labelled-vs-unseen on the
    subzone features (census columns, postcode-level means, sub_n), 5-fold CV AUC (0.5 = exchangeable)
Input: exp/audit/adoption/r2/S_v2.parquet (adopt_v2 subzone table written by b3_nrowtr.py --save-table).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/adoption/r2/b5_split.py
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = "exp/audit/adoption/r2"
S = pd.read_parquet(f"{OUT}/S_v2.parquet")
NTR, NTE = 429303, 183987
NALL = NTR + NTE
lab = S.ad.notna().values
res = {}


def p_all_test(n):
    """hypergeometric probability that all n rows of a subzone fall in the test part"""
    return np.exp(gammaln(NTE + 1) - gammaln(NTE - n + 1) + gammaln(NALL - n + 1) - gammaln(NALL + 1))


tab = []
for n, grp in S.groupby("nrow"):
    if n > 8:
        break
    exp_ = len(grp) * p_all_test(n)
    obs = int((~grp.ad.notna()).sum())
    sd = np.sqrt(len(grp) * p_all_test(n) * (1 - p_all_test(n)))
    tab.append(dict(nrow=int(n), subzones=len(grp), expected_unseen=round(float(exp_), 1), observed_unseen=obs,
                    z=round(float((obs - exp_) / max(sd, 1e-9)), 2)))
res["unseen_count_check"] = tab
print(pd.DataFrame(tab).to_string(index=False))

feats = [c for c in S.columns if c not in ("zone", "cell", "ad", "cr", "nrow", "nrow_tr")]
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, seed=42, deterministic=True, force_row_wise=True, num_threads=2,
              verbose=-1)
for n in (1, 2):
    D = S[S.nrow == n]
    y = (~D.ad.notna()).astype(int).values
    X = D[feats]
    oof = np.zeros(len(D))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), 300)
        oof[b] = m.predict(X.iloc[b])
    auc = roc_auc_score(y, oof)
    # permutation reference: same pipeline with shuffled labels
    yp = np.random.default_rng(0).permutation(y)
    oofp = np.zeros(len(D))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, yp):
        m = lgb.train(params, lgb.Dataset(X.iloc[a], yp[a]), 300)
        oofp[b] = m.predict(X.iloc[b])
    res[f"adv_auc_nrow{n}"] = round(float(auc), 4)
    res[f"adv_auc_nrow{n}_shuffled_ref"] = round(float(roc_auc_score(yp, oofp)), 4)
    res[f"adv_n_nrow{n}"] = [int(len(D)), int(y.sum())]
    print(f"nrow={n}: AUC {auc:.4f} (shuffled-label reference {res[f'adv_auc_nrow{n}_shuffled_ref']}), "
          f"n={len(D)} unseen={y.sum()}", flush=True)
json.dump(res, open(f"{OUT}/b5_split.json", "w"), indent=1)
