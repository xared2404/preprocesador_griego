from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Optional, List

LEX_IN = Path("outputs/lexicon_canonical.jsonl")
SMALL_TSV = Path("outputs/small_words_ud.tsv")

OUT_JSONL = Path("outputs/lexicon_ud.jsonl")
OUT_TSV = Path("outputs/lexicon_ud.tsv")

GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
LATIN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")

def greek_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ς", "σ")
    # keep greek letters and common separators
    out = []
    for ch in s:
        if "α" <= ch <= "ω" or ch in " ,.-=;:/()[]'’":
            out.append(ch)
    return "".join(out).strip()

def head_key(lemma_key: str) -> str:
    """
    Primer átomo útil para clasificar:
    - split por coma, luego por espacio
    - regresa el primer token con griego (ya normalizado)
    """
    k = (lemma_key or "").strip()
    if not k:
        return ""
    # corta por coma primero
    first_chunk = k.split(",", 1)[0].strip()
    # luego por espacios
    for tok in first_chunk.split():
        tok = tok.strip(" .:;()[]'’\"")
        tok2 = greek_key(tok)
        if tok2 and GREEK_RE.search(tok2):
            return tok2
    # fallback: el chunk completo normalizado
    return greek_key(first_chunk)

def feats(**kv) -> str:
    order = ["PronType","Definite","Person","Number","Gender","Case","Polarity"]
    items = []
    for k in order:
        if k in kv and kv[k]:
            items.append(f"{k}={kv[k]}")
    for k in sorted(set(kv.keys()) - set(order)):
        if kv[k]:
            items.append(f"{k}={kv[k]}")
    return "|".join(items)

ARTICLE_KEYS = {"ο", "η", "το", "οι", "αι", "τα"}
CONJ_KEYS = {"και", "η", "αλλα", "δε", "γαρ", "τε", "μεν", "ουν"}
SCONJ_KEYS = {"οτι", "επει", "εαν", "αν", "ει", "οταν"}
ADP_KEYS = {"εν", "εις", "εκ", "απο", "δια", "κατα", "μετα", "υπερ", "επι", "παρα", "προς", "περι", "ανευ", "συν"}
NEG_KEYS = {"ου", "ουκ", "ουχ", "μη", "μηδε", "ουδε"}
PRON_HEADS = {"εγω","συ","ημεις","υμεις","αυτος"}  # heads típicos

def guess_gender_number(entry: dict) -> Tuple[Optional[str], Optional[str]]:
    g = entry.get("gender") or ""
    n = entry.get("number") or ""
    g = g if g in {"M","F","N"} else None
    n = n if n in {"SG","PL"} else None
    return g, n

def ud_from_head(h: str, entry: dict) -> Tuple[Optional[str], str, str]:
    g, n = guess_gender_number(entry)

    if h in ARTICLE_KEYS:
        return ("DET", feats(PronType="Art", Definite="Def", Gender=g, Number=n), "SMALL_WORDS")

    if h in PRON_HEADS:
        if h == "εγω":
            return ("PRON", feats(PronType="Prs", Person="1", Number="SG"), "SMALL_WORDS")
        if h == "συ":
            return ("PRON", feats(PronType="Prs", Person="2", Number="SG"), "SMALL_WORDS")
        if h == "ημεις":
            return ("PRON", feats(PronType="Prs", Person="1", Number="PL"), "SMALL_WORDS")
        if h == "υμεις":
            return ("PRON", feats(PronType="Prs", Person="2", Number="PL"), "SMALL_WORDS")
        return ("PRON", feats(PronType="Prs", Gender=g, Number=n), "SMALL_WORDS")

    if h in CONJ_KEYS:
        return ("CCONJ", "", "SMALL_WORDS")

    if h in SCONJ_KEYS:
        return ("SCONJ", "", "SMALL_WORDS")

    if h in ADP_KEYS:
        return ("ADP", "", "SMALL_WORDS")

    if h in NEG_KEYS:
        return ("PART", feats(Polarity="Neg"), "SMALL_WORDS")

    return (None, "", "")

def load_small_words_index() -> Dict[str, Dict]:
    idx: Dict[str, Dict] = {}
    if not SMALL_TSV.exists():
        return idx
    lines = SMALL_TSV.read_text(encoding="utf-8").splitlines()
    if not lines:
        return idx
    for row in lines[1:]:
        cols = row.split("\t")
        if len(cols) < 4:
            continue
        pdf_page, roman, lemma, gloss = cols[0], cols[1], cols[2], cols[3]
        key = greek_key(lemma)
        if not key:
            continue
        idx[key] = {"roman": roman.strip(), "lemma": lemma, "gloss": gloss}
    return idx

def is_ocr_noise(entry: dict) -> bool:
    """
    Descarta basura obvia de OCR en SMALL_WORDS, especialmente p23:
    - lemma_key sin griego
    - demasiados dígitos/símbolos
    """
    lk = (entry.get("lemma_key") or "").strip()
    ln = (entry.get("lemma_norm") or "").strip()
    s = lk if lk else ln
    if not s:
        return True
    if not GREEK_RE.search(s):
        return True
    # demasiado ruido numérico
    digits = sum(ch.isdigit() for ch in s)
    if digits >= 4:
        return True
    # tokens tipo "=" ">" etc
    if len(s) <= 2:
        return True
    return False

def main():
    if not LEX_IN.exists():
        raise FileNotFoundError(f"No existe: {LEX_IN}")

    small_idx = load_small_words_index()
    out_rows: List[dict] = []

    with LEX_IN.open("r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pdf_page = int(e.get("pdf_page") or 0)
            lemma_key = (e.get("lemma_key") or "").strip()
            lex_class = e.get("lex_class") or e.get("class") or ""

            in_small_window = (lex_class == "SMALL_WORDS") or (22 <= pdf_page <= 35)

            # opcional pero muy recomendable: limpia p23 basura
            if in_small_window and pdf_page == 23 and is_ocr_noise(e):
                continue

            upos = e.get("upos") or ""
            feats_str = e.get("feats") or ""
            lex_subclass = e.get("lex_subclass") or ""

            if in_small_window:
                h = head_key(lemma_key)
                guessed_upos, guessed_feats, guessed_sub = ud_from_head(h, e)
                if guessed_upos:
                    upos = guessed_upos
                if guessed_feats:
                    feats_str = guessed_feats
                if guessed_sub and not lex_subclass:
                    lex_subclass = guessed_sub

                hint = small_idx.get(h) or small_idx.get(lemma_key)  # intenta por head o por lemma_key completo
                if hint and hint.get("roman"):
                    lex_subclass = f"SMALL_WORDS:{hint['roman']}"

            e["upos"] = upos
            e["feats"] = feats_str
            e["lex_subclass"] = lex_subclass
            out_rows.append(e)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for e in out_rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write("section\tpdf_page\tlemma_norm\tlemma_key\tupos\tfeats\tlex_subclass\tgender\tnumber\tgloss_es_clean\tnotes_clean\n")
        for e in out_rows:
            f.write(
                f"{e.get('section','')}\t{e.get('pdf_page','')}\t{e.get('lemma_norm','')}\t{e.get('lemma_key','')}\t"
                f"{e.get('upos','')}\t{e.get('feats','')}\t{e.get('lex_subclass','')}\t"
                f"{e.get('gender','')}\t{e.get('number','')}\t{e.get('gloss_es_clean','')}\t{e.get('notes_clean','')}\n"
            )

    print(f"[OK] wrote: {OUT_JSONL} rows={len(out_rows)}")
    print(f"[OK] wrote: {OUT_TSV}")

if __name__ == "__main__":
    main()
