from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Any

LEXICON = Path("outputs/lexicon_ud.jsonl")
FORMS = Path("outputs/forms_ud.jsonl")

INPUT_TEXT = Path("data/corpus/sample_greek.txt")
OUT_TOKENS = Path("outputs/parsed_tokens.jsonl")
OUT_MD = Path("outputs/annotated_text.md")

GREEK_WORD = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")
PUNCT = set(".,;:!?·—–()[]{}«»\"'’…")

def greek_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ς", "σ")
    return s

def load_lexicon(path: Path) -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            k = (e.get("lemma_key") or "").strip()
            if not k:
                continue
            idx.setdefault(k, []).append(e)
    return idx

def load_forms(path: Path) -> Dict[str, dict]:
    idx: Dict[str, dict] = {}
    if not path.exists():
        return idx
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fk = greek_key(r.get("form",""))
            if fk:
                idx[fk] = r
    return idx

def tokenize(text: str) -> List[Tuple[str,int,int]]:
    toks = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in PUNCT:
            toks.append((ch, i, i+1))
            i += 1
            continue
        m = GREEK_WORD.match(text, i)
        if m:
            toks.append((m.group(0), m.start(), m.end()))
            i = m.end()
        else:
            # fallback: consume one char
            toks.append((ch, i, i+1))
            i += 1
    return toks

def pick_primary(hits: List[dict]) -> dict | None:
    if not hits:
        return None
    # Prefer SMALL_WORDS with upos, else first
    hits2 = sorted(hits, key=lambda x: (0 if (x.get("lex_subclass","").startswith("SMALL_WORDS") and x.get("upos")) else 1))
    return hits2[0]

def annotate(text: str, lex_idx: Dict[str, List[dict]], forms_idx: Dict[str, dict]) -> Tuple[List[dict], str]:
    tokens = tokenize(text)
    records = []
    annotated = []
    last = 0

    for tok, s, e in tokens:
        annotated.append(text[last:s])
        last = e

        if tok in PUNCT:
            annotated.append(tok)
            records.append({"token": tok, "token_key": "", "start": s, "end": e, "hits": [], "hit_count": 0, "primary": {"tag":"PUNCT"}})
            continue

        tok_key = greek_key(tok)

        # 1) FORM override (artículos/pronombres/preps con feats)
        form = forms_idx.get(tok_key)
        if form:
            upos = form.get("upos") or ""
            feats = form.get("feats") or ""
            tag = upos if upos else "∅"
            annotated.append(f"**{tok}**{{{tag}{('|' + feats) if feats else ''}}}")
            records.append({
                "token": tok, "token_key": tok_key, "start": s, "end": e,
                "hits": [], "hit_count": 0,
                "primary": {"tag": tag, "upos": upos, "feats": feats, "source": "forms_ud"}
            })
            continue

        # 2) lexicon lookup (lemma_key exact match)
        hits = lex_idx.get(tok_key, [])
        primary = pick_primary(hits)

        if primary:
            upos = primary.get("upos") or ""
            feats = primary.get("feats") or ""
            tag = upos if upos else "∅"
            annotated.append(f"**{tok}**{{{tag}}}")
            records.append({
                "token": tok, "token_key": tok_key, "start": s, "end": e,
                "hits": hits, "hit_count": len(hits),
                "primary": {"tag": tag, "upos": upos, "feats": feats, "source": "lexicon_ud"}
            })
        else:
            annotated.append(tok)
            records.append({"token": tok, "token_key": tok_key, "start": s, "end": e, "hits": [], "hit_count": 0, "primary": None})

    annotated.append(text[last:])
    return records, "".join(annotated)

def main():
    if not INPUT_TEXT.exists():
        INPUT_TEXT.parent.mkdir(parents=True, exist_ok=True)
        INPUT_TEXT.write_text("ὁ ἄνθρωπος καὶ ἡ γυνή.\nσύ, ἐγώ.\nἐν τῇ οἰκίᾳ.\nεἰς τὸν λόγον.\n", encoding="utf-8")
        print(f"[WARN] No existía {INPUT_TEXT}; creé un sample de prueba.")

    lex_idx = load_lexicon(LEXICON)
    forms_idx = load_forms(FORMS)

    text = INPUT_TEXT.read_text(encoding="utf-8")
    records, md = annotate(text, lex_idx, forms_idx)

    OUT_TOKENS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TOKENS.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    OUT_MD.write_text(md, encoding="utf-8")

    print(f"[OK] tokens: {len(records)}")
    print(f"[OK] wrote: {OUT_TOKENS}")
    print(f"[OK] wrote: {OUT_MD}")
    print(f"[OK] sample input: {INPUT_TEXT}")

if __name__ == "__main__":
    main()
