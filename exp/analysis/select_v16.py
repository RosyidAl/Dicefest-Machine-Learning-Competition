"""v16 member selection: v15nc minus the 17 runtime-pruned members, then greedy forward addition of FB07 members
(only if the honest nested blend score improves), then backward check. Writes exp/analysis/select_v16.json."""
import json, os, re, sys
sys.path.insert(0, "src")
import numpy as np
from blend import cv_blend_score
from common import TARGET_COLS, load, nrmse
log = open("exp/prune_v15nc.log").read()
gas = re.search(r"^gas runs: (\S+)", log, re.M).group(1).split(",")
el = re.search(r"^elec runs: (\S+)", log, re.M).group(1).split(",")
PRUNED = ["l4_lepa", "n6_epm_wM", "p2_gpm", "p3_epm_wM", "lgb_epm_wM_tew", "x4_epm_wM", "la_elec", "la_epm_wM", "x4_lepa",
          "x4_epa", "l4_elec", "xgb_epm_wM", "l4_epm_wM", "p2_lgpm", "la_gpm", "la_epa", "l4_gpm"]
CAND = [c for c in ["xk_gpm", "xk_lgpm", "xk_gpgea", "lk_gpm", "pk_gpm", "xk_lepa", "lk_lepa", "pk_lepm"]
        if os.path.exists(f"exp/oof/{c}.npy")]
ELEC = ("epm", "elec", "lepm", "epa", "lepa")
is_el = lambda r: any(t in r.split("_") for t in ELEC)  # noqa: E731
tr, _ = load(); A = tr.bld_dim_06.values
yg, ye, yl = (tr[c].values for c in (TARGET_COLS[0], TARGET_COLS[2], TARGET_COLS[1]))
P = {}
def col(r):
    if r not in P: P[r] = np.load(f"exp/oof/{r}.npy")
    return P[r]
def fam_score(names, f):
    X = np.column_stack([col(r) for r in names])
    if f == "gas":
        return nrmse(yg, cv_blend_score(X, ye, A) * A, TARGET_COLS[0]) + nrmse(ye, cv_blend_score(X, ye), TARGET_COLS[2])
    return nrmse(yl, cv_blend_score(X, yl), TARGET_COLS[1])
sel = {"gas": [r for r in gas if r not in PRUNED], "elec": [r for r in el if r not in PRUNED]}
cur = {f: fam_score(sel[f], f) for f in sel}
print("pruned base", {f: round(v, 6) for f, v in cur.items()}, "total", round(sum(cur.values()) / 5, 6), flush=True)
for f in ("gas", "elec"):
    pool = [c for c in CAND if is_el(c) == (f == "elec")]
    while pool:
        best = min(((fam_score(sel[f] + [c], f), c) for c in pool))
        if best[0] < cur[f] - 1e-6:
            sel[f].append(best[1]); pool.remove(best[1]); cur[f] = best[0]
            print(f"+{best[1]} {f} {best[0]:.6f}", flush=True)
        else:
            break
    for r in list(sel[f]):  # backward: drop members that no longer help
        s = fam_score([x for x in sel[f] if x != r], f)
        if s <= cur[f] + 1e-7:
            sel[f].remove(r); cur[f] = s; print(f"-{r} {f} {s:.6f}", flush=True)
print("final total", round(sum(cur.values()) / 5, 6), {f: len(v) for f, v in sel.items()})
json.dump({"gas": sel["gas"], "elec": sel["elec"], "score": cur}, open("exp/analysis/select_v16.json", "w"), indent=1)
