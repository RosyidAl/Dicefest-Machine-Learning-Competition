"""honest (nested) blend score of the v15nc gas / elec lists with extra members added (and optional removals)."""
import re, sys
sys.path.insert(0, "src")
import numpy as np
from blend import cv_blend_score
from common import TARGET_COLS, load, nrmse
log = open("exp/prune_v15nc.log").read()
gas = re.search(r"^gas runs: (\S+)", log, re.M).group(1).split(",")
el = re.search(r"^elec runs: (\S+)", log, re.M).group(1).split(",")
add = [a for a in sys.argv[1].split(",") if a] if len(sys.argv) > 1 else []
drop = [a for a in sys.argv[2].split(",") if a] if len(sys.argv) > 2 else []
ELEC = ("epm", "elec", "lepm", "epa", "lepa")
tr, _ = load(); A = tr.bld_dim_06.values
yg, ye, yl = (tr[c].values for c in (TARGET_COLS[0], TARGET_COLS[2], TARGET_COLS[1]))
def sc(g, e):
    G = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in g]); E = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in e])
    s = (nrmse(yg, cv_blend_score(G, ye, A) * A, TARGET_COLS[0]), nrmse(ye, cv_blend_score(G, ye), TARGET_COLS[2]), nrmse(yl, cv_blend_score(E, yl), TARGET_COLS[1]))
    return s, sum(s) / 5
b = sc(gas, el)
g2 = [r for r in gas if r not in drop] + [a for a in add if not any(t in a.split("_") for t in ELEC)]
e2 = [r for r in el if r not in drop] + [a for a in add if any(t in a.split("_") for t in ELEC)]
n = sc(g2, e2)
print("base gas/eui/elec", np.round(b[0], 5), f"{b[1]:.6f}")
print("new  gas/eui/elec", np.round(n[0], 5), f"{n[1]:.6f}", f"delta {n[1] - b[1]:+.6f}")
