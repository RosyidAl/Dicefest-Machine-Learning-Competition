"""Replace the markdown narrative of a notebook with the texts in notebook_text.py, in place.

Code cells and their outputs are left untouched, so an executed notebook stays valid without re-running it.
Usage: python final/patch_notebook_text.py <notebook.ipynb> [...]
"""
import json
import os
import sys

import nbformat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import notebook_text as T  # noqa: E402

# first-line prefix of an existing markdown cell -> new text (old and new headings both match, so reruns are safe)
RULES = [
    (("# DICE FEST 2026",), lambda old: T.INTRO),
    (("## 0. Persiapan",), lambda old: T.SETUP),
    (("## Modul kode",), lambda old: T.MODULES),
    (("## 1. Data",), lambda old: T.DATA),
    (("Sebaran target", "Sebaran kelima target"), lambda old: T.TARGETS_PLOT),
    (("## 2. ",), lambda old: T.FEATURES),
    (("## 3. ",), lambda old: T.ARCH),
    (("## 4. ",), lambda old: T.TRAIN + (T.TRAIN_NOTE if "Catatan untuk salinan" in old else "")),
    (("## 5. ",), lambda old: T.SUBMIT),
    (("## 6. ",), lambda old: T.EVAL),
    (("**Ringkasan galat per target.**",), lambda old: T.EVAL_SUMMARY),
    (("## 7. ",), lambda old: T.POLICY),
    (("Porsi sumbangan", "Porsi kontribusi"), lambda old: T.FAMILIES),
    (("Sepuluh fitur",), lambda old: T.TOP),
    (("Bagaimana sumbangan", "Hubungan antara nilai fitur"), lambda old: T.DEPENDENCE),
    (("### 7.1",), lambda old: T.DRIVERS_MAP),
    (("### 7.3",), lambda old: T.RECS),
    (("## 8. ",), lambda old: T.CLOSING),
]


def as_source(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def patch(path):
    raw = json.load(open(path, encoding="utf-8"))
    nb = nbformat.reads(json.dumps(raw), as_version=4)
    used = set()
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        old = cell.source
        for i, (prefixes, fn) in enumerate(RULES):
            if old.startswith(prefixes):
                assert i not in used, f"two cells match rule {prefixes} in {path}"
                cell.source = fn(old)
                used.add(i)
                break
        else:
            raise SystemExit(f"unmatched markdown cell in {path}: {old[:60]!r}")
    # insert the per-target summary once, right before section 7
    has_summary = any(c.cell_type == "markdown" and c.source.startswith("**Ringkasan galat") for c in nb.cells)
    if not has_summary:
        k = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## 7. "))
        new = nbformat.v4.new_markdown_cell(T.EVAL_SUMMARY)
        new.id = "eval-summary"
        nb.cells.insert(k, new)
    missing = [r[0] for i, r in enumerate(RULES) if i not in used and not r[0][0].startswith("**Ringkasan")]
    assert not missing, f"sections not found in {path}: {missing}"
    nbformat.validate(nb)
    out = json.loads(nbformat.writes(nb))
    for c in out["cells"]:  # keep the list-of-lines layout used by the original file
        if c["cell_type"] == "markdown" and isinstance(c["source"], str):
            c["source"] = as_source(c["source"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"patched {path}: {len(used)} markdown cells replaced, summary {'kept' if has_summary else 'inserted'}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        patch(p)
