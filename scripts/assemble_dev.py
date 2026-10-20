"""Assemble a candidate submission from the development arrays (exp/oof, exp/pred) with the final pipeline's
assemble stage (blend + adoption + residual correction + duplicate copy), as done for sub_v16dev.

The config dir needs members.yaml (only the `blend` lists are read) and resid_knn.json (holdout-fitted weights).
usage: python scripts/assemble_dev.py <config_dir> <work_dir> <out_csv>
"""
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "final", "src"))
import pipeline  # noqa: E402

cfgdir, work, out = sys.argv[1:4]
pipeline.CONF = os.path.abspath(cfgdir)
cfg = pipeline.load_config()
for part in ("oof", "pred"):
    os.makedirs(f"{work}/{part}", exist_ok=True)
for r in sorted(set(cfg["blend"]["gas"] + cfg["blend"]["elec"])):
    for part in ("oof", "pred"):
        dst = f"{work}/{part}/{r}.npy"
        if not os.path.exists(dst):
            os.symlink(os.path.join(ROOT, "exp", part, f"{r}.npy"), dst)
if not os.path.exists(f"{work}/pred/adopt_v2.npy"):
    os.symlink(os.path.join(ROOT, "exp", "pred", "adopt_v2.npy"), f"{work}/pred/adopt_v2.npy")
lamcfg = json.load(open(os.path.join(cfgdir, "resid_knn.json")))
for pl in sorted({p["name"] for v in lamcfg.get("pl", {}).values() for p in v}):  # pseudo-labelled students (PL0001)
    src = os.path.join(ROOT, "exp", "pl", "deploy", f"test_{pl[3:]}.npy")
    assert pl.startswith("pl_") and os.path.exists(src), src
    if not os.path.exists(f"{work}/pred/{pl}.npy"):
        os.symlink(src, f"{work}/pred/{pl}.npy")
logf = open(f"{work}/assemble.log", "a")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


t0 = time.time()
sha = pipeline.assemble(work, out, log=log)
log(f"wrote {out} sha256 {sha} in {(time.time() - t0) / 3600:.2f} h")
