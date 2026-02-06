from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path

OUT_TSV = Path("outputs/forms_ud.tsv")
OUT_JSONL = Path("outputs/forms_ud.jsonl")

GREEK_BLOCK = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")

def strip_diacritics(s: str) -> str:
    # NFD separa letras/diacríticos; eliminamos Mn (marks)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def greek_key(s: str) -> str:
    s = s.strip().lower()
    s = strip_diacritics(s)
    # normaliza sigma final
    s = s.replace("ς", "σ")
    # colapsa espacios
    s = re.sub(r"\s+", " ", s)
    return s

def rec(form: str, lemma_key: str, upos: str, feats: str) -> dict:
    return {
        "form": form,
        "form_key": greek_key(form),
        "lemma_key": lemma_key,
        "upos": upos,
        "feats": feats,
    }

def main():
    rows: list[dict] = []

    # Artículo definido: lemma_key canónico = "ο, η, το" (tu pipeline lo usa así)
    det_lemma = "ο, η, το"

    # Masculino SG
    rows += [
        rec("ὁ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=SG|Case=Nom"),
        rec("τοῦ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=SG|Case=Gen"),
        rec("τῷ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=SG|Case=Dat"),
        rec("τόν", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=SG|Case=Acc"),
    ]
    # Femenino SG
    rows += [
        rec("ἡ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=SG|Case=Nom"),
        rec("τῆς", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=SG|Case=Gen"),
        rec("τῇ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=SG|Case=Dat"),
        rec("τήν", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=SG|Case=Acc"),
    ]
    # Neutro SG
    rows += [
        rec("τό", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=SG|Case=Nom"),
        rec("τοῦ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=SG|Case=Gen"),
        rec("τῷ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=SG|Case=Dat"),
        rec("τό", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=SG|Case=Acc"),
    ]
    # PL
    rows += [
        rec("οἱ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=PL|Case=Nom"),
        rec("τῶν", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=PL|Case=Gen"),
        rec("τοῖς", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=PL|Case=Dat"),
        rec("τούς", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=M|Number=PL|Case=Acc"),

        rec("αἱ", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=PL|Case=Nom"),
        rec("τῶν", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=PL|Case=Gen"),
        rec("ταῖς", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=PL|Case=Dat"),
        rec("τάς", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=F|Number=PL|Case=Acc"),

        rec("τά", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=PL|Case=Nom"),
        rec("τῶν", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=PL|Case=Gen"),
        rec("τοῖς", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=PL|Case=Dat"),
        rec("τά", det_lemma, "DET", "PronType=Art|Definite=Def|Gender=N|Number=PL|Case=Acc"),
    ]

    # Pronombres personales (formas nominativas básicas)
    rows += [
        rec("ἐγώ", "εγω", "PRON", "PronType=Prs|Person=1|Number=SG|Case=Nom"),
        rec("σύ", "συ", "PRON", "PronType=Prs|Person=2|Number=SG|Case=Nom"),
    ]

    # Conjunción coordinante: IMPORTANTÍSIMO incluir ambas variantes (acento agudo vs grave)
    # tok_key de "καὶ" normalmente queda "και"
    rows += [
        rec("καὶ", "και", "CCONJ", "ConjType=Coord"),
        rec("καί", "και", "CCONJ", "ConjType=Coord"),
    ]

    # Preposiciones comunes (mínimo útil)
    rows += [
        rec("ἐν", "εν", "ADP", "ExpectedCase=Dat"),
        rec("εἰς", "εις", "ADP", "ExpectedCase=Acc"),
        rec("ἐκ", "εκ", "ADP", "ExpectedCase=Gen"),
        rec("διά", "δια", "ADP", "ExpectedCase=Gen/Acc"),
        rec("μετά", "μετα", "ADP", "ExpectedCase=Gen/Acc"),
        rec("ὑπέρ", "υπερ", "ADP", "ExpectedCase=Gen/Acc"),
        rec("ἐπί", "επι", "ADP", "ExpectedCase=Gen/Dat/Acc"),
        rec("παρά", "παρα", "ADP", "ExpectedCase=Gen/Dat/Acc"),
        rec("σύν", "συν", "ADP", "ExpectedCase=Dat"),
        rec("ἄνευ", "ανευ", "ADP", "ExpectedCase=Gen"),
    ]

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write("form\tform_key\tlemma_key\tupos\tfeats\n")
        for r in rows:
            f.write(f"{r['form']}\t{r['form_key']}\t{r['lemma_key']}\t{r['upos']}\t{r['feats']}\n")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {OUT_TSV} rows={len(rows)}")
    print(f"[OK] wrote: {OUT_JSONL}")

if __name__ == "__main__":
    main()
