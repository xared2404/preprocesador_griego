# src/preprocesador/normalize.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_GREEK_RANGE_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")  # Greek + Greek Extended
_TOKEN_RE = re.compile(r"[A-Za-z\u0370-\u03FF\u1F00-\u1FFF]+")


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = True
    strip_diacritics: bool = True  # clave para politónico
    normalize_form: str = "NFKD"   # NFKD separa letras + diacríticos


def _strip_marks(s: str) -> str:
    # elimina marcas combinantes (tildes, espíritus, etc.)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def normalize_text(text: str, cfg: NormalizationConfig = NormalizationConfig()) -> str:
    t = unicodedata.normalize(cfg.normalize_form, text)

    if cfg.strip_diacritics and _GREEK_RANGE_RE.search(t):
        t = _strip_marks(t)

    if cfg.lowercase:
        t = t.lower()

    # compacta espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> list[str]:
    # tokens alfabéticos (latín + griego)
    return _TOKEN_RE.findall(text)
