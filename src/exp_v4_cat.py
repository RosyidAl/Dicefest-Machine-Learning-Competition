"""v4 zoo, CatBoost part: 10-fold CV with climate-cell encodings."""
import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data(n_folds=10)
cp = dict(learning_rate=0.08, depth=8, l2_leaf_reg=3, border_count=254)
TE = dict(use_te=True, windows=(2, 10), cells=True, out_dir="exp")
iA, iM = 1.0 / d.A, 1.0 / d.M
run(d, "c4_eui_wA", "cat", "eui", cp, wmul=d.A, **TE)
run(d, "c4_epm_wM", "cat", "epm", cp, wmul=iM, **TE)
run(d, "c4_gpm", "cat", "gpm", cp, wmul=iA, **TE)
run(d, "c4_lepm", "cat", "lepm", cp, wmul=d.M.astype(float), **TE)
