from __future__ import annotations
import re
from pathlib import Path

STRUCT = Path("outputs/vocabulario_structurado.txt")
OUT = Path("outputs/small_words_ud.tsv")

ROMANS = [
    "I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV",
    "XVI","XVII","XVIII","XIX","XX","XXI","XXII","XXIII"
]
ROM_RE = re.compile(r"^(?P<roman>" + "|".join(ROMANS) + r")\b")
PAGE_RE = re.compile(r"^##\s+VOCABULARIO_FUNDAMENTAL\s+—\s+p(?P<p>\d+)\s*$")

GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
LATIN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")

def greek_ratio(s: str) -> float:
    if not s.strip():
        return 0.0
    greek = len(GREEK_RE.findall(s))
    latin = len(LATIN_RE.findall(s))
    digits = sum(ch.isdigit() for ch in s)
    total = len(s)
    # penaliza dígitos/ruido fuerte
    score = greek / max(1, (greek + latin + digits))
    # si casi no hay griego, score cae
    return score

def clean_line(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def split_lemma_gloss(line: str) -> tuple[str,str]:
    # Heurística: primer bloque con griego = lemma, resto = glosa/notas
    # Ej: "ἄμφω ambos L ambo" -> lemma="ἄμφω", gloss="ambos L ambo"
    parts = line.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1].strip()

def main():
    if not STRUCT.exists():
        raise FileNotFoundError(f"No existe: {STRUCT}")

    lines = STRUCT.read_text(encoding="utf-8").splitlines()

    in_small = False
    cur_page = None
    cur_roman = None

    rows = []
    for raw in lines:
        s = raw.rstrip("\n")
        mpage = PAGE_RE.match(s.strip())
        if mpage:
            cur_page = int(mpage.group("p"))
            continue

        if s.strip() == "PALABRAS PEQUEÑAS":
            in_small = True
            cur_roman = None
            continue

        # salimos de SMALL_WORDS cuando dejamos rango 22–35 o cuando entramos a otra sección mayor
        if in_small and cur_page is not None and cur_page > 35:
            break

        if not in_small or cur_page is None:
            continue

        s2 = clean_line(s)
        if not s2:
            continue

        mroman = ROM_RE.match(s2)
        if mroman:
            cur_roman = mroman.group("roman")
            # Puede venir "XVI ..." con texto en la misma línea; lo dejamos caer al siguiente loop si aplica
            rest = s2[len(cur_roman):].strip(" .:-")
            if rest and greek_ratio(rest) >= 0.35 and GREEK_RE.search(rest):
                lemma, gloss = split_lemma_gloss(rest)
                rows.append((cur_page, cur_roman, lemma, gloss))
            continue

        # Filtro anti-OCR-noise
        # Necesitamos algo de griego y ratio suficiente
        if not GREEK_RE.search(s2):
            continue
        if greek_ratio(s2) < 0.35:
            continue

        # Evita líneas tipo "= δ Ξ" o símbolos sueltos
        if len(s2) < 3:
            continue

        lemma, gloss = split_lemma_gloss(s2)
        if not lemma or not GREEK_RE.search(lemma):
            continue

        rows.append((cur_page, cur_roman or "", lemma, gloss))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("pdf_page\troman\tlemma\tgloss\n")
        for p, r, lem, glo in rows:
            f.write(f"{p}\t{r}\t{lem}\t{glo}\n")

    print(f"[OK] wrote: {OUT} rows={len(rows)}")

if __name__ == "__main__":
    main()
