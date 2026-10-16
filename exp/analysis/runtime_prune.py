"""Backward elimination of v15nc blend members by (in-sample loss) / (training hours saved), with honest (nested CV)
scores of the resulting sets. Full-data training times come from the registry (legacy full-data logs only)."""
import re, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from blend import cv_blend_score
from common import SIGMAS, TARGET_COLS, load, nrmse
from prune import gram_nnls

log = open("exp/prune_v15nc.log").read()
gas = re.search(r"^gas runs: (\S+)", log, re.M).group(1).split(",")
el = re.search(r"^elec runs: (\S+)", log, re.M).group(1).split(",")
reg = pd.read_csv("experiments/registry.csv", dtype=str, keep_default_na=False)
T = {r.oof_path.split("/")[-1][:-4]: float(r.runtime_min or 0) / 60 for r in reg.itertuples() if r.oof_path.startswith("exp/oof")}
T["m1_gpm"] = T["m1_epm"] = 3067 / 3600 / 2
tr, _ = load(); A = tr.bld_dim_06.values; n = len(A)
ye, yl, yg = (tr[c].values for c in (TARGET_COLS[2], TARGET_COLS[1], TARGET_COLS[0]))
Pg = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in gas]); PgA = Pg * A[:, None]
Pe = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in el])
st = {"gas": [(Pg.T @ Pg, Pg.T @ ye, ye @ ye, SIGMAS[TARGET_COLS[2]]), (PgA.T @ PgA, PgA.T @ (ye * A), (ye * A) @ (ye * A), SIGMAS[TARGET_COLS[0]])],
      "elec": [(Pe.T @ Pe, Pe.T @ yl, yl @ yl, SIGMAS[TARGET_COLS[1]])]}
names = {"gas": gas, "elec": el}

def ins(fam, S):
    tot = 0
    for G, b, yy, sig in st[fam]:
        w = gram_nnls(G[np.ix_(S, S)], b[S]); tot += np.sqrt(max(yy - 2 * b[S] @ w + w @ G[np.ix_(S, S)] @ w, 0) / n) / sig
    return tot

def honest(Sg, Se):
    G, E = Pg[:, Sg], Pe[:, Se]
    return (nrmse(yg, cv_blend_score(G, ye, A) * A, TARGET_COLS[0]) + nrmse(ye, cv_blend_score(G, ye), TARGET_COLS[2]) + nrmse(yl, cv_blend_score(E, yl), TARGET_COLS[1])) / 5

cur = {"gas": list(range(len(gas))), "elec": list(range(len(el)))}
base_h = honest(cur["gas"], cur["elec"]); saved = 0
pre = sys.argv[1].split(",") if len(sys.argv) > 1 else []
for r in pre:  # resume: members already dropped in an earlier run
    fam = "elec" if r in el else "gas"
    cur[fam] = [x for x in cur[fam] if names[fam][x] != r]; saved += T[r]
if pre:
    print(f"resumed after dropping {pre}: honest {honest(cur['gas'], cur['elec']):.6f} saved {saved:.2f}h", flush=True)
print(f"base honest {base_h:.6f}; total member hours {sum(T[r] for r in gas + el):.2f}", flush=True)
for step in range(16):
    best = None
    for fam in cur:
        c0 = ins(fam, cur[fam])
        for j in cur[fam]:
            r = names[fam][j]
            if r.startswith("m1_"):
                continue
            loss = (ins(fam, [x for x in cur[fam] if x != j]) - c0) / 5
            ratio = loss / max(T[r], 0.01)
            if best is None or ratio < best[0]:
                best = (ratio, fam, j, loss)
    _, fam, j, loss = best
    cur[fam] = [x for x in cur[fam] if x != j]; saved += T[names[fam][j]]
    h = honest(cur["gas"], cur["elec"])
    print(f"drop {names[fam][j]:20s} {T[names[fam][j]]:5.2f}h  honest {h:.6f} ({h - base_h:+.6f})  saved {saved:5.2f}h", flush=True)
