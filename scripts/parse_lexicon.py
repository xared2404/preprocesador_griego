from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Tuple


STRUCTURED_PATH = Path("outputs/vocabulario_structurado.txt")
OUT_TSV = Path("outputs/lexicon.tsv")
OUT_JSONL = Path("outputs/lexicon.jsonl")

ABBREV_MARKER = "LISTA DE ABREVIATURAS Y SIGNOS"
VOCAB_START_PAGE = 22

# PALABRAS PEQUEÑAS: p22–p35 (según tu criterio)
SMALL_WORDS_PAGE_MIN = 22
SMALL_WORDS_PAGE_MAX = 35

HEADER_RE = re.compile(r"^##\s+([A-Z_]+)\s+—\s+p(\d+)\s*$")
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
LATIN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]")
MULTISPACE_RE = re.compile(r"\s{2,}")
PARENS_RE = re.compile(r"\(([^)]+)\)")

SUBENTRY_PREFIX_RE = re.compile(
    r"^(cf\.|cf\b|aor\.|impf\.|fut\.|perf\.|part\.|orig\.|prop\.|ver\b|v\.\b|=|L\b)",
    flags=re.IGNORECASE,
)

OCR_FIXES = [("µ", "μ"), ("ﬁ", "fi"), ("ﬂ", "fl")]

ART_GENDER = {
    "ὁ": ("M", "SG"),
    "ἡ": ("F", "SG"),
    "η": ("F", "SG"),
    "τό": ("N", "SG"),
    "τὸ": ("N", "SG"),
    "το": ("N", "SG"),
    "οἱ": ("M", "PL"),
    "αἱ": ("F", "PL"),
    "τά": ("N", "PL"),
    "τα": ("N", "PL"),
}
ARTICLE_PREFIX_RE = re.compile(r"^(ὁ|ἡ|η|τό|τὸ|το|οἱ|αἱ|τά|τα)\s+(.+)$")

SHORT_LEMMA_WHITELIST = {
    "ὁ", "ἡ", "η", "τό", "τὸ", "το", "οἱ", "αἱ", "τά", "τα",
    "ἐγώ", "σύ", "τί", "τίς", "τι", "τις", "οὗ", "ᾧ", "ὅς", "ἥ", "ὅ",
}

POS_GENDER_TOKENS = {
    "ὁ", "ἡ", "τό", "τὸ", "οἱ", "αἱ", "τά", "το", "τα", "η",
    "adv.", "prep.", "conj.", "interj.", "pron.", "num.",
    "aor.", "fut.", "impf.", "perf.", "part.",
}
POS_GENDER_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(POS_GENDER_TOKENS, key=len, reverse=True)) + r")\b"
)

# Romanos I–XXIII (simple, robusto)
ROMAN_RE = re.compile(r"\b(XXIII|XXII|XXI|XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b")


@dataclass
class LexEntry:
    section: str
    pdf_page: int

    lemma_raw: str
    lemma_norm: str

    pos_gender: str
    gloss_es: str
    notes: str

    gender: str
    number: str
    gender_source: str
    gender_prefix: str
    gender_conflict: int

    # NUEVO
    lex_class: str     # SMALL_WORDS | GENERAL
    roman_group: str   # I..XXIII si se detecta, si no ""

    raw_entry: str


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def clean_ocr(s: str) -> str:
    s = nfc(s)
    for a, b in OCR_FIXES:
        s = s.replace(a, b)
    s = s.replace("\u00A0", " ")
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def norm_lemma(s: str) -> str:
    s = clean_ocr(s)
    s = s.replace("“", "").replace("”", "").replace("«", "").replace("»", "")
    return s


def iter_structured_blocks(text: str) -> Iterable[tuple[str, int, list[str]]]:
    section = None
    page = None
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        h = HEADER_RE.match(line.strip())
        if h:
            if section is not None and page is not None:
                yield section, page, buf
            section = h.group(1)
            page = int(h.group(2))
            buf = []
        else:
            buf.append(line)
    if section is not None and page is not None:
        yield section, page, buf


def extract_abbrev_block(structured_text: str, marker: str) -> list[str]:
    lines = structured_text.splitlines()
    marker_u = marker.upper()

    start_idx = None
    for i, ln in enumerate(lines):
        if marker_u in ln.upper():
            start_idx = i
            break
    if start_idx is None:
        return []

    out = []
    for j in range(start_idx, len(lines)):
        if j != start_idx and lines[j].startswith("## "):
            break
        out.append(clean_ocr(lines[j]))
    return out


def parse_abbrev_map(abbrev_lines: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for ln in abbrev_lines:
        ln = ln.strip()
        if not ln:
            continue
        if ABBREV_MARKER.upper() in ln.upper():
            continue

        m = re.match(r"^([A-Za-zÁÉÍÓÚÑáéíóúñ\.]{1,20})\s*(?:=|:)\s*(.+)$", ln)
        if m:
            mapping[m.group(1).strip()] = m.group(2).strip()
            continue

        m = re.match(r"^([A-Za-zÁÉÍÓÚÑáéíóúñ\.]{1,20})\s+(.+)$", ln)
        if m:
            mapping[m.group(1).strip()] = m.group(2).strip()
            continue

    mapping.setdefault("cf.", "compárese")
    mapping.setdefault("aor.", "aoristo")
    mapping.setdefault("impf.", "imperfecto")
    mapping.setdefault("fut.", "futuro")
    mapping.setdefault("L", "Latín")
    mapping.setdefault("gral.", "en general")

    return mapping


def expand_abbreviations(text: str, abbrev_map: dict[str, str]) -> str:
    if not text or not abbrev_map:
        return text
    out = text
    for key in sorted(abbrev_map.keys(), key=len, reverse=True):
        if f"{key} [" in out:
            continue
        val = abbrev_map[key]
        pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
        out = re.sub(pattern, f"{key} [{val}]", out)
    return out


def looks_like_entry_start(line: str) -> bool:
    return bool(line and GREEK_RE.search(line))


def parse_entries_from_block(lines: list[str]) -> list[str]:
    entries: list[str] = []
    cur: list[str] = []

    for raw in lines:
        line = clean_ocr(raw)
        if not line:
            continue

        if looks_like_entry_start(line):
            if cur:
                entries.append(" ".join(cur).strip())
                cur = []
            cur.append(line)
        else:
            if cur:
                cur.append(line)

    if cur:
        entries.append(" ".join(cur).strip())

    return entries


def infer_gender_from_pos(pos_gender: str) -> Tuple[str, str]:
    pg = (pos_gender or "").strip()
    if pg == "τὸ":
        pg = "τό"
    if pg == "το":
        pg = "τό"
    if pg == "τα":
        pg = "τά"
    if pg == "η":
        pg = "ἡ"
    if pg in ART_GENDER:
        return ART_GENDER[pg]
    return ("", "")


def detect_article_prefix(lemma_raw: str) -> Tuple[str, str, str, str]:
    lr = lemma_raw.strip()
    m = ARTICLE_PREFIX_RE.match(lr)
    if not m:
        return (lr, "", "", "")
    art = m.group(1).strip()
    rest = m.group(2).strip()
    if not rest or not GREEK_RE.search(rest):
        return (lr, "", "", "")
    gender, number = ART_GENDER.get(art, ("", ""))
    return (rest, gender, number, art)


def is_det_set_entry(lemma_raw: str) -> bool:
    lr = lemma_raw.strip().replace("τὸ", "τό")
    lr = re.sub(r"\s+", " ", lr)
    return ("ὁ" in lr) and (("ἡ" in lr) or ("η" in lr)) and ("τό" in lr)


def extract_roman_group(raw_entry: str) -> str:
    m = ROMAN_RE.search(raw_entry or "")
    return m.group(1) if m else ""


def split_entry(entry_text: str, abbrev_map: dict[str, str]) -> tuple[str, str, str, str, str, str, str, str, int]:
    text = clean_ocr(entry_text)

    notes_parts = [p.strip() for p in PARENS_RE.findall(text)]
    body = PARENS_RE.sub(" ", text)
    body = MULTISPACE_RE.sub(" ", body).strip()

    chunks = [c.strip() for c in re.split(r"\s*\|\s*|\s*;\s*", body) if c.strip()]

    main_chunks = []
    for c in chunks:
        if SUBENTRY_PREFIX_RE.match(c):
            notes_parts.append(c)
        else:
            main_chunks.append(c)

    main = " ".join(main_chunks).strip() if main_chunks else body

    pos_gender = ""
    mpos = POS_GENDER_RE.search(main)
    if mpos and mpos.start() <= 60:
        pos_gender = mpos.group(1)

    anchor = None
    for pat in [r"\s(el|la|los|las)\s", r"[A-Za-zÁÉÍÓÚÑáéíóúñ]"]:
        m = re.search(pat, main)
        if m:
            anchor = m.start()
            break
    if anchor is None or anchor <= 0:
        for pat in [":", ", "]:
            k = main.find(pat)
            if k > 0:
                anchor = k
                break

    if anchor is None or anchor <= 0:
        lemma_raw = main.strip()
        gloss = ""
    else:
        lemma_raw = main[:anchor].strip(" :;,")
        gloss = main[anchor:].strip(" :;,").lstrip()

    lemma_raw = re.sub(r"^\d{1,4}\s+", "", lemma_raw).strip()

    mlat = LATIN_RE.search(lemma_raw)
    if mlat:
        lemma_raw = lemma_raw[:mlat.start()].strip(" :;,")

    gender = ""
    number = ""
    gender_source = "none"
    gender_prefix = ""
    gender_conflict = 0

    if is_det_set_entry(lemma_raw):
        gender = "DET_SET"
        gender_source = "det_set"
    else:
        lemma_no_art, g1, n1, pref = detect_article_prefix(lemma_raw)
        if g1:
            lemma_raw = lemma_no_art
            gender, number, gender_source, gender_prefix = g1, n1, "prefix", pref

            g2, _ = infer_gender_from_pos(pos_gender)
            if g2 and g2 != gender:
                gender_conflict = 1
        else:
            g2, n2 = infer_gender_from_pos(pos_gender)
            if g2:
                gender, number, gender_source = g2, n2, "pos_gender"

    gloss = expand_abbreviations(gloss.strip(), abbrev_map)
    notes = " | ".join([clean_ocr(x) for x in notes_parts if x.strip()])
    notes = expand_abbreviations(notes, abbrev_map)

    return lemma_raw.strip(), pos_gender, gloss, notes, gender, number, gender_source, gender_prefix, gender_conflict


def is_bad_short_lemma(lemma: str) -> bool:
    lemma = (lemma or "").strip()
    if not lemma:
        return True
    if len(lemma) <= 1:
        return lemma not in SHORT_LEMMA_WHITELIST
    return False


def main() -> None:
    print("[DEBUG] parse_lexicon v7 running", flush=True)

    if not STRUCTURED_PATH.exists():
        raise FileNotFoundError(f"No encuentro: {STRUCTURED_PATH.resolve()}")

    structured = STRUCTURED_PATH.read_text(encoding="utf-8")

    abbrev_lines = extract_abbrev_block(structured, ABBREV_MARKER)
    abbrev_map = parse_abbrev_map(abbrev_lines)
    print(f"[DEBUG] abbrev lines: {len(abbrev_lines)}", flush=True)
    print(f"[DEBUG] abbrev map size: {len(abbrev_map)}", flush=True)

    out: list[LexEntry] = []

    for section, page, lines in iter_structured_blocks(structured):
        if page < VOCAB_START_PAGE:
            continue
        if section in ("FRONT_MATTER", "INDICE_ALFABETICO"):
            continue

        lex_class = "SMALL_WORDS" if (SMALL_WORDS_PAGE_MIN <= page <= SMALL_WORDS_PAGE_MAX) else "GENERAL"

        entries = parse_entries_from_block(lines)
        for e in entries:
            lemma_raw, pos_gender, gloss, notes, gender, number, gender_source, gender_prefix, gender_conflict = split_entry(e, abbrev_map)

            if not lemma_raw:
                continue
            if not GREEK_RE.search(lemma_raw):
                continue
            if is_bad_short_lemma(lemma_raw):
                continue

            raw_clean = clean_ocr(e)
            roman_group = extract_roman_group(raw_clean) if lex_class == "SMALL_WORDS" else ""

            out.append(
                LexEntry(
                    section=section,
                    pdf_page=page,
                    lemma_raw=lemma_raw,
                    lemma_norm=norm_lemma(lemma_raw),
                    pos_gender=pos_gender,
                    gloss_es=gloss,
                    notes=notes,
                    gender=gender,
                    number=number,
                    gender_source=gender_source,
                    gender_prefix=gender_prefix,
                    gender_conflict=gender_conflict,
                    lex_class=lex_class,
                    roman_group=roman_group,
                    raw_entry=raw_clean,
                )
            )

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write(
            "section\tpdf_page\tlemma_raw\tlemma_norm\tpos_gender\tgloss_es\tnotes\t"
            "gender\tnumber\tgender_source\tgender_prefix\tgender_conflict\t"
            "lex_class\troman_group\traw_entry\n"
        )
        for it in out:
            f.write(
                f"{it.section}\t{it.pdf_page}\t{it.lemma_raw}\t{it.lemma_norm}\t{it.pos_gender}\t"
                f"{it.gloss_es}\t{it.notes}\t{it.gender}\t{it.number}\t{it.gender_source}\t"
                f"{it.gender_prefix}\t{it.gender_conflict}\t{it.lex_class}\t{it.roman_group}\t{it.raw_entry}\n"
            )

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    print(f"[OK] entries: {len(out)}", flush=True)
    print(f"[OK] wrote: {OUT_TSV}", flush=True)
    print(f"[OK] wrote: {OUT_JSONL}", flush=True)


if __name__ == "__main__":
    main()
