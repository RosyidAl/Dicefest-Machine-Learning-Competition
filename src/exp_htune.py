"""Hyper-parameter variants judged on the holdout split of exp_hold.py (fold-averaged H predictions behave like
test predictions, so higher-variance / lower-bias settings that look worse on OOF can win there).
Outputs go to exp/hold/tune/{oof,pred}; compare with the base member in exp/hold/pred.

usage: python src/exp_htune.py gpu|cpu
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from exp_hold import CPU, GPU, TE, W, s  # noqa: E402
from train_lib import run  # noqa: E402

# name, base member, parameter overrides
VARIANTS = [
    ("xt_gpm_mcw10", "x4_gpm", dict(min_child_weight=10)),
    ("xt_gpm_cs4", "x4_gpm", dict(colsample_bytree=0.4)),
    ("xt_gpm_ss6", "x4_gpm", dict(subsample=0.6)),
    ("xt_gpm_d10", "x4_gpm", dict(max_depth=10)),
    ("xt_gpm_lr03", "x4_gpm", dict(learning_rate=0.03)),
    ("xt_gpm_d12", "x4_gpm", dict(max_depth=12)),
    ("xt_gpm_d10_lr03", "x4_gpm", dict(max_depth=10, learning_rate=0.03)),
    ("xt_lepa_d10", "x4_lepa", dict(max_depth=10)),
    ("xt_gpgea_d10", "x4_gpgea", dict(max_depth=10)),
    ("lt_gpm_nl1023", "l4_gpm", dict(num_leaves=1023, min_data_in_leaf=50)),
    ("xt_gpgea_d10_lr03", "x4_gpgea", dict(max_depth=10, learning_rate=0.03)),
    ("xt_lgpm_d10_lr03", "x4_lgpm", dict(max_depth=10, learning_rate=0.03)),
    ("xt_gas_d10_lr03", "x4_gas", dict(max_depth=10, learning_rate=0.03)),
    ("xt_eui_wA_d10_lr03", "x4_eui_wA", dict(max_depth=10, learning_rate=0.03)),
    ("xt_gpn_d10_lr05", "x4_gpn", dict(max_depth=10, learning_rate=0.05)),
    ("ct_gpm_d10", "c4_gpm", dict(depth=10)),
    ("lt_gpm_lr02", "l4_gpm", dict(learning_rate=0.02)),
    ("lt_gpm_et", "l4_gpm", dict(extra_trees=True)),
    ("lt_epa_et", "l4_epa", dict(extra_trees=True)),
    ("lt_gpm_lin", "l4_gpm", dict(linear_tree=True, linear_lambda=1.0)),
]
spec = {r[0]: r for r in GPU + CPU}
for name, base, over in VARIANTS:
    _, backend, kind, params, wk = spec[base]
    if (backend == "lgb") != (sys.argv[1] == "cpu") or os.path.exists(f"exp/hold/tune/pred/{name}.npy"):
        continue
    run(s, name, backend, kind, dict(params, **over), wmul=W[wk], verbose=False, **dict(TE, out_dir="exp/hold/tune"))
