"""Compare experiments with their parent on the folds both evaluated (MASTER_PROMPT 6.2 acceptance rule).

usage: python scripts/compare.py E0201 E0202 ...      (parent from the registry, or --parent E0200)
Prints a markdown table: per target NRMSE of child and parent on the common folds, delta, folds improved, SE of the
per-fold delta, and the verdict (DITERIMA if delta < 0 and (improved >= 4/5 of folds or |delta| > 2 SE)).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tracking import read_registry  # noqa: E402


def metrics(eid):
    reg = read_registry().set_index("exp_id")
    out = reg.loc[eid, "oof_path"]
    if not out:
        return None
    path = os.path.join(os.path.dirname(out), "metrics.json")
    return json.load(open(path)) if os.path.exists(path) else None


def compare(child, parent):
    mc, mp = metrics(child), metrics(parent)
    rows = []
    if mc is None or mp is None:
        return rows
    for t, c in mc["per_target"].items():
        if t not in mp["per_target"]:
            continue
        fc, fp = np.array(c["fold_nrmse"]), np.array(mp["per_target"][t]["fold_nrmse"])
        n = min(len(fc), len(fp))
        d = fc[:n] - fp[:n]
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        imp = int((d < 0).sum())
        ok = d.mean() < 0 and (imp >= np.ceil(0.8 * n) or abs(d.mean()) > 2 * se)
        rows.append(dict(target=t.replace("target_", ""), child=fc[:n].mean(), parent=fp[:n].mean(), delta=d.mean(),
                         improved=f"{imp}/{n}", se=se, verdict="DITERIMA" if ok else "DITOLAK"))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--parent")
    a = ap.parse_args()
    reg = read_registry().set_index("exp_id")
    print("| Eksperimen | Parent | Target | NRMSE | Parent | Delta | Fold membaik | SE | Verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in a.ids:
        p = a.parent or reg.loc[e, "parent_id"]
        if not p:
            continue
        for r in compare(e, p):
            print(f"| {e} {reg.loc[e, 'hypothesis'][:45]} | {p} | {r['target']} | {r['child']:.5f} | {r['parent']:.5f} | "
                  f"{r['delta']:+.5f} | {r['improved']} | {r['se']:.5f} | {r['verdict']} |")
