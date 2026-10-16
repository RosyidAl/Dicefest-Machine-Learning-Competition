"""Submission validator (MASTER_PROMPT Lampiran G). Appends a row to submissions/submission_log.csv unless --no-log.

usage: python scripts/validate_submission.py <file.csv> [--source S] [--cv X] [--note N] [--sample] [--no-log]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from metric import TARGET_COLS  # noqa: E402
from tracking import now_wib, sha256_file  # noqa: E402

HEADER = ["id"] + TARGET_COLS
LOG = "submissions/submission_log.csv"
LOG_COLS = ["created_wib", "file", "sha256", "source", "cv_mean", "public_lb", "note"]


def validate(path, sample=False):
    sub = pd.read_csv(path, float_precision="round_trip")
    ref = pd.read_csv("data/raw/sample_submission.csv")
    errors, warns = [], []
    if list(sub.columns) != HEADER:
        errors.append(f"header {list(sub.columns)}")
    if len(sub) != 183987:
        errors.append(f"rows {len(sub)}")
    if len(sub) != len(ref) or not (sub["id"].astype(str).values == ref["id"].astype(str).values).all():
        errors.append("id column differs from sample_submission (values or order)")
    vals = sub[TARGET_COLS]
    if not all(pd.api.types.is_numeric_dtype(vals[c]) for c in TARGET_COLS) or not np.isfinite(vals.values).all():
        errors.append("non-numeric, NaN or inf values")
    elif not sample:
        for c in TARGET_COLS[:3]:
            if (vals[c] < 0).any():
                errors.append(f"{c} negative")
        for c in TARGET_COLS[3:]:
            if ((vals[c] < 0) | (vals[c] > 100)).any():
                errors.append(f"{c} outside 0..100")
        tr = pd.read_parquet("data/cache/train.parquet", columns=TARGET_COLS)
        for c in TARGET_COLS:
            d = abs(vals[c].mean() / tr[c].mean() - 1)
            if d > 0.10:
                warns.append(f"{c}: mean {vals[c].mean():.4g} vs train {tr[c].mean():.4g} ({d:.1%})")
    return errors, warns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--source", default="")
    ap.add_argument("--cv", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--sample", action="store_true", help="format checks only (sample_submission)")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    errors, warns = validate(a.file, a.sample)
    sha = sha256_file(a.file)
    for w in warns:
        print("WARN", w)
    if errors:
        print("INVALID", a.file, *errors, sep="\n  ")
        sys.exit(1)
    print(f"VALID {a.file} sha256 {sha}")
    if not a.no_log:
        log = pd.read_csv(LOG, dtype=str, keep_default_na=False) if os.path.exists(LOG) else pd.DataFrame(columns=LOG_COLS)
        if not ((log["file"] == a.file) & (log["sha256"] == sha)).any():
            row = dict(created_wib=now_wib(), file=a.file, sha256=sha, source=a.source, cv_mean=a.cv, public_lb="",
                       note=a.note)
            pd.concat([log, pd.DataFrame([row])], ignore_index=True)[LOG_COLS].to_csv(LOG, index=False)
