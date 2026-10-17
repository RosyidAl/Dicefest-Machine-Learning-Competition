"""v17 member selection: the frozen v16 blend (final/configs/members.yaml), then greedy forward addition of the
Poisson members of src/exp_v17_pois.py (only if the honest nested blend score improves), then backward check.
Writes exp/analysis/select_v17.json."""
import json, os, sys
sys.path.insert(0, "src")
import numpy as np
import yaml
from blend import cv_blend_score
from common import TARGET_COLS, load, nrmse
cfg = yaml.safe_load(open("final/configs/members.yaml"))
CAND = [c for c in ["xp_gpm", "xp_eui", "xp_epm", "xp_gpgea", "xp_epa", "xp8_gpm", "lp_gpm", "lp_epm"] if os.path.exists(f"exp/oof/{c}.npy")]
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
for c in CAND:  # single-member scores
    y, nm = (yl, TARGET_COLS[1]) if is_el(c) else (ye, TARGET_COLS[2])
    extra = "" if is_el(c) else f" gas {nrmse(yg, col(c) * A, TARGET_COLS[0]):.5f}"
    print(f"single {c}: {nrmse(y, col(c), nm):.5f}{extra}", flush=True)
sel = {"gas": list(cfg["blend"]["gas"]), "elec": list(cfg["blend"]["elec"])}
cur = {f: fam_score(sel[f], f) for f in sel}
print("v16 base", {f: round(v, 6) for f, v in cur.items()}, "total", round(sum(cur.values()) / 5, 6), flush=True)
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
json.dump({"gas": sel["gas"], "elec": sel["elec"], "score": cur}, open("exp/analysis/select_v17.json", "w"), indent=1)
