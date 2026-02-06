from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict


IN_TSV = Path("outputs/lexicon.tsv")
OUT_JSONL = Path("outputs/lexicon_canonical.jsonl")
OUT_TSV = Path("outputs/lexicon_canonical.tsv")

TAB = "\t"
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
FORM_TAG_RE = re.compile(r"\b(aor\.|impf\.|fut\.|perf\.|part\.)\b", flags=re.IGNORECASE)


@dataclass
class CanonEntry:
    section: str
    pdf_page: int
    lemma_norm: str
    lemma_key: str
    lemma_key_ascii: str

    gender: str
    number: str
    gender_source: str

    pos_gender: str
    gloss_es_clean: str
    notes_clean: str
    forms: List[str]

    # NUEVO
    lex_class: str
    roman_group: str


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def strip_brackets_expansions(s: str) -> str:
    if not s:
        return s
    return re.sub(r"\s*\[[^\]]+\]", "", s).strip()

def greek_strip_diacritics(s: str) -> str:
    s = nfc(s)
    decomposed = unicodedata.normalize("NFD", s)
    out = []
    for ch in decomposed:
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))

def greek_key(s: str) -> str:
    s = greek_strip_diacritics(s)
    s = s.lower()
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def greek_key_ascii(s: str) -> str:
    base = greek_key(s)
    return "".join(ch if ord(ch) < 128 else "" for ch in base)

def extract_forms(notes: str) -> List[str]:
    if not notes:
        return []
    forms = []
    seen = set()
    for m in FORM_TAG_RE.finditer(notes):
        tag = m.group(1).lower()
        if tag not in seen:
            seen.add(tag)
            forms.append(tag)
    return forms

def read_tsv(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError("TSV vacío")
    header = lines[0].split(TAB)

    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split(TAB)
        if len(parts) > len(header):
            parts = parts[:len(header)-1] + [TAB.join(parts[len(header)-1:])]
        if len(parts) < len(header):
            parts = parts + [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return rows


def main():
    if not IN_TSV.exists():
        raise FileNotFoundError(f"No encuentro: {IN_TSV.resolve()}")

    rows = read_tsv(IN_TSV)

    out: List[CanonEntry] = []
    for r in rows:
        lemma_norm = nfc(r.get("lemma_norm", "").strip())
        if not lemma_norm or not GREEK_RE.search(lemma_norm):
            continue

        gloss_es = (r.get("gloss_es", "") or "").strip()
        notes = (r.get("notes", "") or "").strip()

        gloss_clean = strip_brackets_expansions(gloss_es)
        notes_clean = strip_brackets_expansions(notes)

        forms = extract_forms(notes)

        out.append(
            CanonEntry(
                section=r.get("section","").strip(),
                pdf_page=int(r.get("pdf_page","0") or 0),
                lemma_norm=lemma_norm,
                lemma_key=greek_key(lemma_norm),
                lemma_key_ascii=greek_key_ascii(lemma_norm),
                gender=r.get("gender","").strip(),
                number=r.get("number","").strip(),
                gender_source=r.get("gender_source","").strip(),
                pos_gender=r.get("pos_gender","").strip(),
                gloss_es_clean=gloss_clean,
                notes_clean=notes_clean,
                forms=forms,
                lex_class=r.get("lex_class","").strip() or "GENERAL",
                roman_group=r.get("roman_group","").strip(),
            )
        )

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write(
            "section\tpdf_page\tlemma_norm\tlemma_key\tgender\tnumber\tgender_source\t"
            "lex_class\troman_group\tpos_gender\tgloss_es_clean\tnotes_clean\tforms\n"
        )
        for e in out:
            f.write(
                f"{e.section}\t{e.pdf_page}\t{e.lemma_norm}\t{e.lemma_key}\t{e.gender}\t{e.number}\t{e.gender_source}\t"
                f"{e.lex_class}\t{e.roman_group}\t{e.pos_gender}\t{e.gloss_es_clean}\t{e.notes_clean}\t{'|'.join(e.forms)}\n"
            )

    print(f"[OK] canonical entries: {len(out)}")
    print(f"[OK] wrote: {OUT_JSONL}")
    print(f"[OK] wrote: {OUT_TSV}")


if __name__ == "__main__":
    main()
