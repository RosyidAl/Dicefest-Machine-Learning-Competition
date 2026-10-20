"""Integrity audit 1: submission file format vs sample_submission, and CSV float precision.

Reads the files as text (no float parsing) for ids/headers, then parses the floats with round_trip and checks that
the decimal text written for every value is the shortest repr that round-trips (i.e. no precision was lost).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_format.py [submission.csv]
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

SUB = sys.argv[1] if len(sys.argv) > 1 else "submissions/sub_v19vce.csv"
SS = "data/raw/sample_submission.csv"
TARGETS = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm",
           "target_clean_building_ratio", "target_electric_heating_adoption"]
out = {}

raw_sub = open(SUB, "rb").read()
out["sha256"] = hashlib.sha256(raw_sub).hexdigest()
out["crlf"] = raw_sub.count(b"\r\n")
out["bom"] = raw_sub[:3] == b"\xef\xbb\xbf"
out["ends_with_newline"] = raw_sub.endswith(b"\n")

s_txt = pd.read_csv(SUB, dtype=str, keep_default_na=False)
r_txt = pd.read_csv(SS, dtype=str, keep_default_na=False)
out["header_equal"] = list(s_txt.columns) == list(r_txt.columns)
out["rows"] = [len(s_txt), len(r_txt)]
out["id_equal_text"] = bool(len(s_txt) == len(r_txt) and (s_txt["id"].values == r_txt["id"].values).all())
out["id_unique"] = bool(s_txt["id"].is_unique)
out["empty_cells"] = int((s_txt[TARGETS] == "").sum().sum())

# the test parquet used by the pipeline must have the same id order
te_id = pd.read_parquet("data/cache/test.parquet", columns=["id"])["id"].astype(str).values
out["id_equal_test_parquet"] = bool((te_id == s_txt["id"].values).all())
te_raw_id = pd.read_csv("data/raw/test.csv", usecols=["id"], dtype=str)["id"].values
out["id_equal_test_csv"] = bool((te_raw_id == s_txt["id"].values).all())

sub = pd.read_csv(SUB, float_precision="round_trip")
out["dtypes"] = {c: str(sub[c].dtype) for c in sub.columns}
v = sub[TARGETS].values
out["all_finite"] = bool(np.isfinite(v).all())
out["min"] = {c: float(sub[c].min()) for c in TARGETS}
out["max"] = {c: float(sub[c].max()) for c in TARGETS}

# precision: text written must equal repr(float(text)) (shortest round-trip); count digits
prec = {}
for c in TARGETS:
    txt = s_txt[c].values
    vals = sub[c].values
    rt = np.array([repr(float(t)) for t in txt])
    same = (rt == txt)
    # values that would change if the text were re-parsed with the default (non round-trip) C parser
    fast = pd.read_csv(SUB, usecols=[c])[c].values
    sig = np.array([len(t.replace("-", "").replace(".", "").lstrip("0")) for t in txt])
    prec[c] = dict(text_is_shortest_repr=float(same.mean()), median_sig_digits=float(np.median(sig)),
                   min_sig_digits=int(sig.min()),
                   default_parser_max_rel_diff=float(np.max(np.abs(fast - vals) / np.maximum(np.abs(vals), 1e-300))),
                   default_parser_max_abs_diff_over_sigma=float(np.max(np.abs(fast - vals))))
out["precision"] = prec
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_format.json", "w"), indent=1)
