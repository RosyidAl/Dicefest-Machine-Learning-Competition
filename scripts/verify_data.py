"""Phase 0 data check: shapes, column names, per-prefix counts, md5 of the raw files, cache equality."""
import hashlib
import json

import pandas as pd

PREFIXES = ["bld_dim_", "bld_era_", "hvac_dist_", "env_climate_", "socio_demog_"]
res = {}
for name in ["train", "test", "sample_submission"]:
    h = hashlib.md5()
    with open(f"data/raw/{name}.csv", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    raw = pd.read_csv(f"data/raw/{name}.csv", engine="pyarrow")
    cache = pd.read_parquet(f"data/cache/{name}.parquet")
    same = raw.shape == cache.shape and list(raw.columns) == list(cache.columns) and all(
        raw[c].equals(cache[c]) for c in raw.columns)
    cnt = {p: sum(c.startswith(p) for c in raw.columns) for p in PREFIXES}
    res[name] = {"md5": h.hexdigest(), "shape": list(raw.shape), "prefix_counts": cnt,
                 "n_features": sum(cnt.values()), "other_cols": [c for c in raw.columns if not c.startswith(tuple(PREFIXES))],
                 "cache_identical": bool(same), "dtypes": raw.dtypes.astype(str).value_counts().to_dict()}
    print(name, res[name]["shape"], res[name]["md5"], cnt, "cache_identical", same)
assert res["train"]["shape"] == [429303, 209] and res["test"]["shape"] == [183987, 204]
assert res["sample_submission"]["shape"] == [183987, 6]
json.dump(res, open("env/data_check.json", "w"), indent=1)
