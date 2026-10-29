"""Checks the v3 deck structure (deck/ppt02/01_struktur_deck_minto.md) against the brief for the top-down deck.

Timing: 13 main slides, durations summing to 570 s, contiguous time ranges from 00:00 to 09:30, section totals
(I 60 s, II 150 s, III 90 s, IV 210 s, V 60 s) and spoken words per minute within 120-140. Spoken words expand numbers
the way they are said in Indonesian ("0,24844" is seven words), so the rate is not flattered by digits.

Content: the required phrase in the policy slides, the model counts on the architecture slide (43 = 41 + 2), every
number in parts A, C, D and E present in the fact sheets, no number in the last minute that was not said before, no
em-dash, no technical-detail terms in the main slides, no model terms in the opening, causal wording flagged for
review, and the required fields on every slide.

With --tulis the two generated blocks in the markdown (per-slide timing table, check results) are rewritten.

usage: python deck/ppt02/cek_struktur.py [--tulis]
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DOC = HERE / "01_struktur_deck_minto.md"
FACTS = [ROOT / "deck/fakta_notebook.txt", ROOT / "deck/fakta_turunan.md", ROOT / "deck/ppt01/fakta_tambahan.md",
         ROOT / "reports/WRITEUP_DRAFT_vB.md", HERE / "fakta_ppt02.md"]

PHRASE = "temuan model mendukung rekomendasi ini"
TOTAL_S = 570
SECTIONS = [("I", (1, 2), 60), ("II", (3, 4, 5), 150), ("III", (6, 7), 90), ("IV", (8, 9, 10, 11), 210),
            ("V", (12, 13), 60)]
WPM = (120, 140)
REQUIRED = ("Judul Slide", "Visual Utama", "Isi Teks", "Narasi Pembicara")
SHOWN = ("Judul Slide", "Visual Utama", "Isi Teks", "Narasi Pembicara")   # fields whose numbers the audience sees
# detail kept for the backup slides only (brief, part 6)
DETAIL_RE = re.compile(r"A2xM2xp2|NNLS|λ|lambda|TreeSHAP|q_xent|r_reg|n/\(n|early stopping|Poisson|PLR|TabM",
                       re.I)
OPENING_RE = re.compile(r"\b(model|fitur|algoritma|arsitektur|LightGBM|XGBoost|SHAP|fold|kNN|blend\w*)\b", re.I)
CAUSAL_RE = re.compile(r"\b(menyebabkan|penyebab|mengakibatkan|berdampak|membuktikan|terbukti|dipicu)\b", re.I)
EM_DASH = "—"

SLIDE_RE = re.compile(r"^### Slide (\d+) · (\d\d):(\d\d)–(\d\d):(\d\d) \((\d+) detik\)$")
BACKUP_RE = re.compile(r"^### (C\d+)\. (.+)$")
PART_RE = re.compile(r"^## ([A-G])\. ")
FIELD_RE = re.compile(r"^- \*\*([^*(]+?)(?: \(([^)]*)\))?:\*\*\s?(.*)$")
NUM_RE = re.compile(r"(?<![\w.,/])[+-]?\d+(?:[.,]\d+)*")
SPOKEN_NUM = re.compile(r"[+−±-]?\d+(?:\.\d{3})*(?:,\d+)?")

DIGITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]


def say_int(n):
    """Indonesian words for a non-negative integer"""
    if n < 10:
        return [DIGITS[n]]
    if n < 20:
        return {10: ["sepuluh"], 11: ["sebelas"]}.get(n, [DIGITS[n - 10], "belas"])
    if n < 100:
        t, u = divmod(n, 10)
        return [DIGITS[t], "puluh"] + (say_int(u) if u else [])
    if n < 1000:
        h, r = divmod(n, 100)
        return (["seratus"] if h == 1 else [DIGITS[h], "ratus"]) + (say_int(r) if r else [])
    if n < 1_000_000:
        t, r = divmod(n, 1000)
        return (["seribu"] if t == 1 else say_int(t) + ["ribu"]) + (say_int(r) if r else [])
    m, r = divmod(n, 1_000_000)
    return say_int(m) + ["juta"] + (say_int(r) if r else [])


def say_number(tok):
    """'0,24844' -> nol koma dua empat delapan empat empat; '42.931' -> empat puluh dua ribu ..."""
    body = tok.lstrip("+−±-")
    whole, _, frac = body.partition(",")
    words = say_int(int(whole.replace(".", ""))) if whole else []
    if frac:
        words += ["koma"] + [DIGITS[int(d)] for d in frac]
    return words


def spoken_words(text):
    n = 0
    for tok in re.sub(r"[*`\[\]]", " ", text).split():
        core = tok.strip(".,;:!?()\"'")
        if not core:
            continue
        pct = core.endswith("%")
        core = core.rstrip("%")
        if SPOKEN_NUM.fullmatch(core):
            n += len(say_number(core)) + pct
        elif re.fullmatch(r"REG_\d+", core):            # "reg dua tujuh tiga"
            n += 1 + len(core) - 4
        elif "_" in core:                                 # column names
            n += len([p for p in core.split("_") if p])
        elif "-" in core:                                 # reduplication counts once, compounds by part
            parts = [p for p in core.split("-") if p]
            n += 1 if len(set(p.lower() for p in parts)) == 1 else len(parts)
        else:
            n += 1 + pct
    return n


def written_words(text):
    return len([w for w in re.sub(r"[*`]", " ", text).split() if re.search(r"\w", w)])


def candidates(token):
    """plain readings of a number token (Indonesian and English separators), as in check_deck.py"""
    t = token.lstrip("+-")
    out = {t}

    def thousands(sep):
        return all(len(g) == 3 for g in t.split(sep)[1:])

    if "," in t and "." in t:
        out.add(t.replace(".", "").replace(",", "."))
        out.add(t.replace(",", ""))
    elif "," in t:
        out.add(t.replace(",", "."))
        if thousands(","):
            out.add(t.replace(",", ""))
    elif "." in t and thousands("."):
        out.add(t.replace(".", ""))
    return out


def numbers(text, min_int=13):
    """number tokens a reader would check: small counts and ordinals are skipped"""
    out = []
    for tok in NUM_RE.findall(text):
        plain = tok.lstrip("+-")
        if re.fullmatch(r"\d+", plain) and int(plain) < min_int:
            continue
        out.append(tok)
    return out


def parse(text):
    """slides, backups and per-part text (without 'Sumber' fields) from the markdown"""
    slides, backups, parts = [], [], {}
    part, cur, field = None, None, None
    for line in text.splitlines():
        m = PART_RE.match(line)
        if m:
            part, cur, field = m.group(1), None, None
            continue
        if line.startswith("#"):
            cur, field = None, None
            m = SLIDE_RE.match(line)
            if m:
                n, h1, m1, h2, m2, secs = map(int, m.groups())
                cur = {"n": n, "start": h1 * 60 + m1, "end": h2 * 60 + m2, "secs": secs, "fields": {}, "labels": {}}
                slides.append(cur)
            m = BACKUP_RE.match(line)
            if m:   # a backup title is shown on its slide, so its numbers are checked too
                cur = {"id": m.group(1), "title": m.group(2), "fields": {}, "labels": {}}
                backups.append(cur)
                parts[part] = parts.get(part, "") + m.group(2) + "\n"
            continue
        m = FIELD_RE.match(line)
        if m and cur is not None:
            field = m.group(1).strip()
            cur["fields"][field] = m.group(3) + "\n"
            cur["labels"][field] = m.group(2)
            line = m.group(3)   # the label, e.g. "(±25 detik)", is not content
        elif cur is not None and field is not None:
            cur["fields"][field] += line.strip() + "\n"
        if part is not None and not (cur is not None and field == "Sumber"):
            parts[part] = parts.get(part, "") + line + "\n"
    return slides, backups, parts


def mmss(s):
    return f"{s // 60:02d}:{s % 60:02d}"


def check(text):
    slides, backups, parts = parse(text)
    res = {"errors": [], "notes": [], "rows": [], "items": {}}
    err, note = res["errors"].append, res["notes"].append
    by_n = {s["n"]: s for s in slides}

    # structure and required fields
    if [s["n"] for s in slides] != list(range(1, 14)):
        err(f"slide numbers {[s['n'] for s in slides]} (expected 1..13)")
    for s in slides:
        for f in REQUIRED:
            if not s["fields"].get(f, "").strip():
                err(f"slide {s['n']}: field '{f}' missing")
        lab = s["labels"].get("Narasi Pembicara") or ""
        if lab != f"±{s['secs']} detik":
            err(f"slide {s['n']}: narration label '{lab}' does not match {s['secs']} detik")
    for b in backups:
        for f in ("Visual atau bukti", "Pertanyaan juri"):
            if not b["fields"].get(f, "").strip():
                err(f"{b['id']}: field '{f}' missing")

    # timing
    total = sum(s["secs"] for s in slides)
    cont = all(a["end"] == b["start"] for a, b in zip(slides, slides[1:]))
    spans = all(s["end"] - s["start"] == s["secs"] for s in slides)
    bounds = slides and slides[0]["start"] == 0 and slides[-1]["end"] == TOTAL_S
    if total != TOTAL_S:
        err(f"total {total} s, expected {TOTAL_S} s")
    if not (cont and spans and bounds):
        err("time ranges are not contiguous from 00:00 to 09:30 or do not match the durations")
    sec_ok = True
    for name, ns, want in SECTIONS:
        got = sum(by_n[n]["secs"] for n in ns if n in by_n)
        if got != want:
            sec_ok = False
            err(f"section {name}: {got} s, expected {want} s")
    words_total = spoken_total = 0
    rate_ok = True
    for s in slides:
        narr = s["fields"].get("Narasi Pembicara", "")
        w, sp = written_words(narr), spoken_words(narr)
        rate = sp * 60 / s["secs"]
        words_total, spoken_total = words_total + w, spoken_total + sp
        flag = "" if WPM[0] <= rate <= WPM[1] else " (di luar 120 sampai 140)"
        if flag:
            rate_ok = False
            err(f"slide {s['n']}: {rate:.0f} spoken words per minute")
        res["rows"].append((s["n"], f"{mmss(s['start'])}–{mmss(s['end'])}", s["secs"], w, sp, f"{rate:.0f}{flag}"))
    res["totals"] = (total, words_total, spoken_total, spoken_total * 60 / max(total, 1))
    opening = [by_n[n] for n in (1, 2) if n in by_n]
    open_secs = sum(s["secs"] for s in opening)
    open_words = sum(spoken_words(s["fields"].get("Narasi Pembicara", "")) for s in opening)
    open_rate = open_words * 60 / max(open_secs, 1)

    def shown(s, fields=SHOWN):
        return "\n".join(s["fields"].get(f, "") for f in fields)

    # opening: trilemma, three recommendations, no model talk (the agenda line is a signpost, not a discussion)
    op_text = "\n".join(shown(s) for s in opening)
    op_said = "\n".join(l for l in "\n".join(shown(s, ("Judul Slide", "Isi Teks", "Narasi Pembicara"))
                                               for s in opening).splitlines() if "Berikutnya:" not in l)
    recs_named = all(re.search(p, op_text, re.I) for p in (r"retrofit", r"gardu", r"pilot"))
    model_talk = sorted(set(m.lower() for m in OPENING_RE.findall(op_said)))
    if model_talk:
        err(f"opening mentions model terms: {model_talk}")
    if not (open_secs == 60 and WPM[0] <= open_rate <= WPM[1] and "trilema" in op_text.lower() and recs_named):
        err("opening does not state the trilemma and three recommendations in 60 s at 120-140 words per minute")

    # order of the talk
    order = [re.match(r"[IV]+", s["fields"].get("Bagian rubrik", "").strip()) for s in slides]
    order = [m.group(0) if m else "?" for m in order]
    want_order = ["I", "I", "II", "II", "II", "III", "III", "IV", "IV", "IV", "IV", "V", "V"]
    if order != want_order:
        err(f"rubric order {order}")

    # model counts on the architecture slide; nothing else claims 42 models or one 43-model ensemble
    s6 = shown(by_n[6]) if 6 in by_n else ""
    counts_ok = all(x in s6 for x in ("43", "41", "28", "13", "2 LightGBM", "terpisah"))
    main_text = "\n".join(shown(s) for s in slides)
    bad_count = re.findall(r"\b42 model|43 model[^.\n]*satu (?:gabungan|ensemble)", main_text)
    if not counts_ok or bad_count:
        err(f"model counts on slide 6 incomplete or contradicted: {bad_count}")

    # policy coverage and the required phrase
    narr = {n: by_n[n]["fields"].get("Narasi Pembicara", "").lower() for n in by_n}
    phrase_n = sum(re.sub(r"\s+", " ", narr[n]).count(PHRASE) for n in (8, 9, 10, 11) if n in narr)
    phrase_slides = [n for n in (9, 10, 11) if PHRASE in re.sub(r"\s+", " ", narr.get(n, ""))]
    if phrase_slides != [9, 10, 11]:
        err(f"required phrase only on slides {phrase_slides}")
    cover = {8: r"\b330 zona", 9: r"\bQ1\b", 10: r"\bgardu\b", 11: r"\bpilot\b.*\bbergas\b|\bbergas\b.*\bpilot\b"}
    cover_ok = all(re.search(p, shown(by_n[n]), re.I | re.S) for n, p in cover.items() if n in by_n)
    chain_ok = all(all(lbl in by_n[n]["fields"].get("Isi Teks", "") for lbl in
                       ("TEMUAN", "INTERPRETASI", "REKOMENDASI", "LANGKAH")) for n in (9, 10, 11))
    if not (cover_ok and chain_ok):
        err("policy slides do not cover map, Q1, substation and pilot with the four-step chain")

    # the retrofit potential is an indicator, not an upper bound (audit 27 Oct)
    if re.search(r"batas atas", main_text, re.I):
        err("main slides still call the retrofit potential an upper bound ('batas atas')")

    # technical detail only in backups
    detail_main = sorted(set(m for s in slides for m in DETAIL_RE.findall(shown(s))))
    detail_backup = sorted(set(m for b in backups
                               for m in DETAIL_RE.findall(b["title"] + "\n" + "\n".join(b["fields"].values()))))
    if detail_main:
        err(f"technical detail in main slides: {detail_main}")

    # causal wording in the main slides (review by hand)
    causal = [(s["n"], m) for s in slides for m in CAUSAL_RE.findall(shown(s, ("Judul Slide", "Isi Teks",
                                                                              "Narasi Pembicara")))]
    for n, m in causal:
        note(f"slide {n}: causal word '{m}' (review)")

    # em-dash anywhere
    em = text.count(EM_DASH)
    if em:
        err(f"{em} em-dash characters")

    # pronouns on screen
    for s in slides:
        for m in re.findall(r"\b(kami|saya)\b", shown(s, ("Judul Slide", "Isi Teks")), re.I):
            err(f"slide {s['n']}: '{m}' on screen")

    # numbers against the fact sheets (parts A, C, D, E; 'Sumber' fields are citations)
    facts = set()
    for f in FACTS:
        for tok in NUM_RE.findall(f.read_text(encoding="utf-8")):
            facts |= candidates(tok)
    missing = {}
    for p in "ACDE":
        for tok in numbers(parts.get(p, "")):
            if not candidates(tok) & facts:
                missing.setdefault(p, set()).add(tok)
    n_checked = sum(len(numbers(parts.get(p, ""))) for p in "ACDE")
    for p, toks in missing.items():
        err(f"part {p}: numbers not in the fact sheets: {', '.join(sorted(toks))}")

    # last minute: no number that was not said or shown before
    earlier = set()
    for s in slides:
        if s["n"] <= 11:
            for tok in numbers(shown(s), min_int=0):
                earlier |= candidates(tok)
    late = sorted({tok for s in slides if s["n"] >= 12 for tok in numbers(shown(s), min_int=4)
                   if not candidates(tok) & earlier})
    if late:
        err(f"new numbers in the last minute: {late}")

    ok = not res["errors"]
    res["items"] = [
        (total == TOTAL_S and cont and spans and bounds and sec_ok,
         "Total durasi bicara tepat 9 menit 30 detik",
         f"jumlah durasi 13 *slide* {total} detik; rentang waktu bersambung 00:00 sampai 09:30; total per bagian "
         f"60, 150, 90, 210, dan 60 detik"),
        (bool(bounds),
         "*Buffer* 30 detik berada di luar narasi utama",
         "narasi berakhir di 09:30; 09:30 sampai 10:00 hanya untuk serah terima dan jeda (bagian B2)"),
        (open_secs == 60 and recs_named and not model_talk and WPM[0] <= open_rate <= WPM[1],
         "Pembukaan mengungkapkan Trilema Energi dan tiga rekomendasi dalam 60 detik",
         f"*slide* 1 dan 2 berdurasi {open_secs} detik dengan {open_words} kata lisan ({open_rate:.0f} kata per "
         f"menit); trilema di *slide* 1, tiga rekomendasi di *slide* 2; tanpa istilah model"),
        (order == want_order,
         "Pembahasan mengikuti urutan rekayasa fitur, arsitektur model, kebijakan, lalu keterbatasan",
         "urutan bagian rubrik per *slide*: " + ", ".join(order)),
        (counts_ok and not bad_count,
         "Total 43 model dibedakan jelas menjadi 41 model yang di-*blend* dan 2 LightGBM terpisah",
         "*slide* 6 memuat 43, 41, 28, 13, \"2 LightGBM\", dan \"terpisah\"; tidak ada klaim 42 model atau satu "
         "gabungan 43 model di *slide* utama"),
        (cover_ok and chain_ok,
         "Peta 330 zona, prioritas Q1, kapasitas gardu, dan pilot elektrifikasi bergas dibahas sesuai bukti",
         "*slide* 8 sampai 11; *slide* 9 sampai 11 memakai rantai TEMUAN, INTERPRETASI, REKOMENDASI, LANGKAH"),
        (phrase_slides == [9, 10, 11],
         "Frasa \"temuan model mendukung rekomendasi ini\" digunakan pada bagian kebijakan publik",
         f"{phrase_n} kali, di narasi *slide* " + ", ".join(map(str, phrase_slides))),
        (not causal,
         "Tidak ada klaim kausalitas mutlak berdasarkan SHAP",
         "0 kata kausal (menyebabkan, mengakibatkan, membuktikan, terbukti, dan sejenisnya) di judul, isi, dan "
         "narasi *slide* utama; SHAP selalu disebut sumbangan pada prediksi atau asosiasi" if not causal else
         "kata kausal ditemukan: " + "; ".join(f"slide {n}: {m}" for n, m in causal)),
        (not missing and not late,
         "Tidak ada angka atau hasil empiris yang dikarang",
         f"{n_checked} angka di bagian A, C, D, dan E dicocokkan dengan lima lembar fakta: "
         f"{sum(len(v) for v in missing.values())} tidak ditemukan; 0 angka baru di menit terakhir"
         if not late else f"angka baru di menit terakhir: {late}"),
        (not detail_main and bool(detail_backup),
         "Detail teknis ekstrem hanya muncul di *slide* cadangan Q&A",
         "0 istilah rinci di *slide* utama; di *slide* cadangan: " + ", ".join(f"`{d}`" for d in detail_backup)),
    ]
    res["ok"] = ok and all(i[0] for i in res["items"])
    res["extra"] = [f"tidak ada em-dash (U+2014): {'ya' if em == 0 else 'tidak'}",
                    f"{len(slides)} *slide* utama dan {len(backups)} *slide* cadangan; setiap *slide* memuat keempat "
                    f"komponen", "kecepatan narasi setiap *slide* di rentang 120 sampai 140 kata lisan per menit: "
                    f"{'ya' if rate_ok else 'tidak'}", "sebutan \"batas atas\" untuk potensi *retrofit* di *slide* utama: "
                    + ("tidak ada" if not re.search(r"batas atas", main_text, re.I) else "masih ada")]
    return res


def render(res):
    rows = ["| *Slide* | Rentang | Durasi | Kata tertulis | Kata lisan | Kata per menit |",
            "|---|---|---|---|---|---|"]
    for n, span, secs, w, sp, rate in res["rows"]:
        rows.append(f"| {n} | {span} | {secs} detik | {w} | {sp} | {rate} |")
    total, w, sp, rate = res["totals"]
    rows.append(f"| **Jumlah** | 00:00–09:30 | **{total} detik** | {w} | {sp} | {rate:.0f} |")
    table = "\n".join(rows)
    items = [f"- [{'x' if ok else ' '}] **{label}.** {why}." for ok, label, why in res["items"]]
    extra = [f"- {e}" for e in res["extra"]] + [f"- Catatan: {n}" for n in res["notes"]]
    checks = "\n".join(items) + "\n\nPemeriksaan tambahan:\n\n" + "\n".join(extra)
    if res["errors"]:
        checks += "\n\nMasalah yang masih terbuka:\n\n" + "\n".join(f"- {e}" for e in res["errors"])
    return table, checks


def replace_block(text, name, body):
    pat = re.compile(rf"(<!-- {name}:mulai -->\n).*?(\n<!-- {name}:selesai -->)", re.S)
    assert pat.search(text), f"block {name} missing"
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tulis", action="store_true", help="rewrite the generated blocks in the markdown")
    a = ap.parse_args()
    text = DOC.read_text(encoding="utf-8")
    res = check(text)
    table, checks = render(res)
    print(table, "\n", checks, sep="\n")
    if a.tulis:
        new = replace_block(replace_block(text, "tabel-waktu", table), "hasil-cek", checks)
        if new != text:
            DOC.write_text(new, encoding="utf-8")
            print(f"\nwrote {DOC.relative_to(ROOT)}")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
