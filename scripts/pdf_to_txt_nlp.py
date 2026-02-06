from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path


# =========================
# CONFIGURACIÓN
# =========================

PDF_PATH = Path(
    "data/corpus/raw/vocabulario_griego/"
    "Vocabulario-Fundamental-y-Constructivo-Del-Griego.pdf"
)

OUT_STRUCT = Path("outputs/vocabulario_structurado.txt")
OUT_NLP = Path("outputs/vocabulario_nlp_ready.txt")

LANG = "spa+ell"
DPI = 250  # sube a 300–400 si quieres más precisión en griego

START_MARKER = "PALABRAS PEQUEÑAS"
VOCAB_LAST_PDF_PAGE = 237  # última página útil antes del índice (respaldo duro)

SECTION_RULES = [
    ("VOCABULARIO_FUNDAMENTAL", r"\bVOCABULARIO\s+FUNDAMENTAL\b"),
    ("VOCABULARIO_CONSTRUCTIVO", r"\bVOCABULARIO\s+CONSTRUCTIVO\b"),
    ("INDICE_ALFABETICO", r"\b[IÍ]NDICE\s+ALFAB[ÉE]TICO\b"),
]


# =========================
# REGEX / PARSE
# =========================

HEADER_RE = re.compile(r"^##\s+([A-Z_]+)\s+—\s+p(\d+)\s*$")


# =========================
# UTILIDADES
# =========================

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    return text


def cleanup_page(text: str) -> str:
    text = normalize_unicode(text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # une palabras cortadas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str, current: str) -> str:
    t = text.upper()
    for name, pattern in SECTION_RULES:
        if re.search(pattern, t):
            return name
    return current


def to_nlp_ready(structured: str) -> str:
    lines = []
    in_front = False

    for raw in structured.splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith("## FRONT_MATTER"):
            in_front = True
            continue

        if line.startswith("## "):
            in_front = False
            lines.append(line)
            continue

        if in_front:
            continue

        if re.fullmatch(r"\d{1,4}", line):
            continue

        line = re.sub(r"\s{2,}", " ", line)
        if len(line) < 3:
            continue

        lines.append(line)

    return "\n".join(lines)


def find_vocab_start_page_from_marker(structured_text: str, marker: str) -> int | None:
    """
    Encuentra la línea donde aparece el marcador (REAL en el TXT),
    busca hacia atrás el header más cercano y devuelve su pN.
    """
    lines = structured_text.splitlines()

    marker_idx = None
    marker_u = marker.upper()
    for idx, line in enumerate(lines):
        if marker_u in line.upper():
            marker_idx = idx
            break

    if marker_idx is None:
        return None

    # Busca el header anterior más cercano
    for j in range(marker_idx, -1, -1):
        m = HEADER_RE.match(lines[j].strip())
        if m:
            return int(m.group(2))  # pN
    return None


def relabel_headers_before_page(structured_text: str, vocab_start_page: int) -> str:
    """
    Re-etiqueta como FRONT_MATTER todos los headers con p < vocab_start_page.
    Conserva p == vocab_start_page (la página del marcador) como vocabulario.
    """
    out_lines = []
    for raw in structured_text.splitlines():
        m = HEADER_RE.match(raw.strip())
        if m:
            page = int(m.group(2))
            if page < vocab_start_page:
                out_lines.append(f"## FRONT_MATTER — p{page}")
            else:
                out_lines.append(raw)
        else:
            out_lines.append(raw)
    return "\n".join(out_lines)


# =========================
# MAIN
# =========================

def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"No encuentro el PDF: {PDF_PATH.resolve()}")

    OUT_STRUCT.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Convirtiendo PDF a imágenes…", flush=True)
    pages = convert_from_path(str(PDF_PATH), dpi=DPI)

    OUT_STRUCT.write_text("", encoding="utf-8")

    vocab_started = False
    current_section = "FRONT_MATTER"

    for i, img in enumerate(pages, start=1):
        if i > VOCAB_LAST_PDF_PAGE:
            print(f"[STOP] Alcancé VOCAB_LAST_PDF_PAGE={VOCAB_LAST_PDF_PAGE}.", flush=True)
            break

        print(f"[OCR] p{i:03d} …", flush=True)

        text = pytesseract.image_to_string(
            img,
            lang=LANG,
            config="--oem 1 --psm 6"
        )
        text = cleanup_page(text)
        text_upper = text.upper()

        # Antes del marcador: SIEMPRE FRONT_MATTER
        if not vocab_started:
            current_section = "FRONT_MATTER"
            if START_MARKER in text_upper:
                vocab_started = True
                current_section = "VOCABULARIO_FUNDAMENTAL"
        else:
            maybe = detect_section(text, current_section)
            if maybe == "INDICE_ALFABETICO":
                print("[STOP] Detecté INDICE_ALFABETICO. Cortando antes del índice.", flush=True)
                break
            current_section = maybe

        chunk = f"## {current_section} — p{i}\n{text}\n\n"
        with OUT_STRUCT.open("a", encoding="utf-8") as f:
            f.write(chunk)

        print(f"[OK] p{i:03d} -> {current_section}", flush=True)

    # ========= POST-PROCESO DEFINITIVO =========
    structured = OUT_STRUCT.read_text(encoding="utf-8")

    vocab_start_page = find_vocab_start_page_from_marker(structured, START_MARKER)
    if vocab_start_page is not None:
        structured = relabel_headers_before_page(structured, vocab_start_page)
        OUT_STRUCT.write_text(structured, encoding="utf-8")
        print(f"[FIX] Re-etiqueté todo p < {vocab_start_page} como FRONT_MATTER.", flush=True)
    else:
        print("[WARN] No encontré el marcador; no pude re-etiquetar retroactivamente.", flush=True)

    # NLP-ready
    nlp = to_nlp_ready(structured)
    OUT_NLP.write_text(nlp, encoding="utf-8")

    print("\nListo:", flush=True)
    print(f"- {OUT_STRUCT}", flush=True)
    print(f"- {OUT_NLP}", flush=True)


if __name__ == "__main__":
    main()
