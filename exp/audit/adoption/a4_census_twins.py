"""Adoption lens, check 4: do different subzone ids share an identical census vector (same output area listed twice)?

If an unseen test subzone had a labelled twin with the same hvac_dist + socio_demog vector, its adoption could be
copied. Also: are the subzone-level census vectors unique, and do labelled twins agree on adoption?
usage: .venv/bin/python exp/audit/adoption/a4_census_twins.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import HVAC  # noqa: E402

OUT = "exp/audit/adoption"
S = pd.read_parquet(f"{OUT}/sz.parquet")
const = json.load(open(f"{OUT}/const_cols.json"))["const_cols"]
lab = S.ad.notna().values
for name, cols in (("hvac only", HVAC), ("census all const", const),
                   ("socio only", [c for c in const if c.startswith("socio")])):
    key = pd.util.hash_pandas_object(S[cols].round(9), index=False).values
    vc = pd.Series(key).value_counts()
    dupkey = set(vc[vc > 1].index)
    isdup = np.array([k in dupkey for k in key])
    print(f"[{name}] subzones in a shared-vector group: {isdup.sum()} (labelled {int((isdup & lab).sum())}, "
          f"unlabelled {int((isdup & ~lab).sum())}); groups {len(dupkey)}")
    if name != "hvac only":
        df = pd.DataFrame({"k": key, "ad": S.ad.values, "lab": lab, "nrow": S.nrow.values})[isdup]
        g = df.groupby("k")
        both = g.lab.agg(lambda x: x.any() and (~x).any())
        nl = g.ad.count()
        agree = g.ad.agg(lambda x: x.dropna().nunique() <= 1)
        print(f"   groups with labelled+unlabelled: {int(both.sum())}; groups with >=2 labelled: {int((nl >= 2).sum())},"
              f" of which adoption agrees: {int(agree[nl >= 2].sum())}")
        # unlabelled rows that could be copied
        cand = df[~df.lab].k.isin(both[both].index)
        print(f"   unlabelled subzones with a labelled twin: {int(cand.sum())}, their test rows:"
              f" {int(df[~df.lab][cand.values].nrow.sum())}")
