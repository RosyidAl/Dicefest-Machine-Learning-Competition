"""Exports the slide content of 01_struktur_deck_v4.md to build/isi_v4.json for build_deck_v4.js.

The markdown stays the single source of the deck text: titles, on-screen text ("Isi Teks"), speaker notes ("Narasi
Pembicara"), timing, sources, the notebook cells printed in the footer ("Sel notebook"), and for the backup slides the
evidence blocks, the jury questions and the answer guide. Inline markup (*italic*, `code`, **bold**, ^superscript^) is
kept and turned into text runs by the builder.

Titles that need two lines are split where both lines are about equally long (Carlito Bold metrics, the same as
Calibri, at the title size of each layout), preferring a break after a comma or colon and avoiding a line that ends
in a preposition or conjunction. PowerPoint then never breaks inside a word such as "out-of-fold".

Each field becomes a list of blocks:
  {"jenis": "butir", "tingkat": 0, "teks": "..."}   bullet (level from the indentation)
  {"jenis": "paragraf", "teks": "..."}                text outside a bullet
  {"jenis": "tabel", "baris": [[cell, ...], ...]}     table, header first, separator row dropped

usage: .venv/bin/python deck/ppt03/isi_deck.py   (writes deck/ppt03/build/isi_v4.json; needs Pillow)
"""
import json
import re
from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
DOC = HERE / "01_struktur_deck_v4.md"
OUT = HERE / "build" / "isi_v4.json"

SLIDE_RE = re.compile(r"^### Slide (\d+) · (\d\d:\d\d)–(\d\d:\d\d) \((\d+) detik\)$")
BACKUP_RE = re.compile(r"^### (C\d+)\. (.+)$")
FIELD_RE = re.compile(r"^- \*\*([^*(]+?)(?: \(([^)]*)\))?:\*\*\s?(.*)$")
SPEAKER_ROW = re.compile(r"^\| (I|II|III|IV|V)\. [^|]+\| ([\d, ]+) \|[^|]+\|[^|]+\| ([^|]+) \|$")
FONTS = HERE.parent / "ppt01" / "fonts"
TITLE_W = 12.133                                   # title placeholder width in build_deck_v4.js (inches)
TITLE_PT = {"GELAP": 32, "ISI": 28, "CADANGAN": 26}  # title size per layout
WEAK_END = {"di", "ke", "dari", "dan", "yang", "per", "pada", "untuk", "dengan", "oleh", "atau", "dalam", "bukan"}
_BOLD = ImageFont.truetype(str(FONTS / "Carlito-Bold.ttf"), 1000)
_BOLD_IT = ImageFont.truetype(str(FONTS / "Carlito-BoldItalic.ttf"), 1000)


def title_lines(title, pt, clauses=False):
    """one line, or two balanced lines, of a title in markup; with clauses=True one line per clause after the
    colon when every clause fits (the closing slide recaps the three actions this way)"""
    words = title.split(" ")

    def width(ws):   # inches; italic markup words measured in the italic face, kerning ignored (wider, so safer)
        units = sum((_BOLD_IT if w.startswith("*") and not w.startswith("**") else _BOLD).getlength(re.sub(r"[*`]", "", w))
                    for w in ws) + _BOLD.getlength(" ") * (len(ws) - 1)
        return units / 1000 * pt / 72

    limit = TITLE_W * 0.97
    if width(words) <= limit:
        return [title]
    if clauses:
        head, tail = title.split(": ", 1)
        parts = tail.split(", ")
        lines = [f"{head}: {parts[0]},"] + [p + "," for p in parts[1:-1]] + [parts[-1]]
        if len(lines) == 3 and all(width(ln.split(" ")) <= limit for ln in lines):
            return lines
    best = None
    for k in range(1, len(words)):
        w1, w2 = width(words[:k]), width(words[k:])
        if w1 > limit or w2 > limit:
            continue
        last = re.sub(r"[*`]", "", words[k - 1])
        cost = max(w1, w2) + (0.02 * limit if w2 > w1 else 0)
        if last.endswith((",", ":")) and min(k, len(words) - k) >= 3:
            cost -= 0.25 * limit
        elif last.lower() in WEAK_END:
            cost += 0.1 * limit
        if best is None or cost < best[0]:
            best = (cost, k)
    assert best, f"title does not fit on two lines: {title}"
    return [" ".join(words[:best[1]]), " ".join(words[best[1]:])]


def blocks(lines):
    """field lines (indentation kept) -> blocks"""
    out, cur = [], None
    for raw in lines:
        line = raw.rstrip()
        s = line.strip()
        if not s:
            cur = None
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not out or out[-1]["jenis"] != "tabel":
                out.append({"jenis": "tabel", "baris": []})
            out[-1]["baris"].append(cells)
            cur = None
            continue
        if s.startswith("- "):
            level = (len(line) - len(line.lstrip())) // 2
            cur = {"jenis": "butir", "tingkat": level, "teks": s[2:]}
            out.append(cur)
            continue
        if cur is not None:
            cur["teks"] += " " + s
        else:
            cur = {"jenis": "paragraf", "teks": s}
            out.append(cur)
    return out


def plain(lines):
    return " ".join(ln.strip() for ln in lines if ln.strip())


def parse(text):
    main, backup, speakers, answerer = [], [], {}, {}
    cur, field, table_mode = None, None, None
    for line in text.splitlines():
        if line.startswith("## "):
            cur, field = None, None
            table_mode = None
            continue
        if line.startswith("### "):
            cur, field = None, None
            table_mode = "penjawab" if line.strip() == "### Pembagian penjawab" else None
            m = SLIDE_RE.match(line)
            if m:
                cur = {"n": int(m.group(1)), "mulai": m.group(2), "selesai": m.group(3), "detik": int(m.group(4)),
                       "medan": {}}
                main.append(cur)
            m = BACKUP_RE.match(line)
            if m:
                cur = {"id": m.group(1), "judul": m.group(2), "medan": {}}
                backup.append(cur)
            continue
        m = SPEAKER_ROW.match(line)
        if m:
            for n in m.group(2).split(","):
                speakers[int(n)] = m.group(3).strip()
            continue
        if table_mode == "penjawab" and line.startswith("|") and "C" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            for cid in re.findall(r"C\d+", cells[1]):
                answerer[cid] = cells[2]
            continue
        if cur is None:
            continue
        m = FIELD_RE.match(line)
        if m and not line.startswith(" "):
            field = m.group(1).strip()
            cur["medan"][field] = [m.group(3)] if m.group(3) else []
            continue
        if field is not None:
            cur["medan"][field].append(line[2:] if line.startswith("  ") else line)
    return main, backup, speakers, answerer


def main():
    text = DOC.read_text(encoding="utf-8")
    main_raw, backup_raw, speakers, answerer = parse(text)
    assert [s["n"] for s in main_raw] == list(range(1, 14)), [s["n"] for s in main_raw]
    assert [b["id"] for b in backup_raw] == [f"C{i}" for i in range(1, 12)], [b["id"] for b in backup_raw]
    utama = []
    for s in main_raw:
        f = s["medan"]
        utama.append({"n": s["n"], "mulai": s["mulai"], "selesai": s["selesai"], "detik": s["detik"],
                      "bagian": plain(f["Bagian rubrik"]), "pembicara": speakers[s["n"]],
                      "judul": plain(f["Judul Slide"]),
                      "judul_baris": title_lines(plain(f["Judul Slide"]), TITLE_PT["GELAP" if s["n"] in (1, 13) else "ISI"],
                                                 clauses=s["n"] == 13),
                      "visual": plain(f["Visual Utama"]),
                      "isi": blocks(f["Isi Teks"]), "narasi": plain(f["Narasi Pembicara"]),
                      "sumber": plain(f["Sumber"]), "sel": plain(f["Sel notebook"])})
    cadangan = []
    for b in backup_raw:
        f = b["medan"]
        questions = [q.strip().strip('"') for q in plain(f["Pertanyaan juri"]).split(" · ")]
        cadangan.append({"id": b["id"], "judul": b["judul"], "judul_baris": title_lines(b["judul"], TITLE_PT["CADANGAN"]),
                         "bukti": blocks(f["Visual atau bukti"]),
                         "pertanyaan": questions, "pedoman": plain(f["Pedoman jawaban"]),
                         "sumber": plain(f["Sumber"]), "penjawab": answerer.get(b["id"], "")})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"utama": utama, "cadangan": cadangan}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE.parent.parent)}: {len(utama)} main slides, {len(cadangan)} backups")


if __name__ == "__main__":
    main()
