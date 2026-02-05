from __future__ import annotations

from functools import lru_cache
from typing import List

from .normalize import normalize_text, tokenize


@lru_cache(maxsize=1)
def _get_lemmatizer():
    """
    Construye una sola vez el lematizador (CLTK puede ser pesado).
    Requiere que ya hayas importado: grc_models_cltk
    """
    from cltk.lemmatize.grc import GreekBackoffLemmatizer
    return GreekBackoffLemmatizer()


def lemmatize_greek(text: str) -> List[str]:
    """
    Devuelve una lista de lemas.
    Si algo falla (por entorno/modelos), devuelve tokens normalizados (fallback).
    """
    t = normalize_text(text)
    toks = tokenize(t)

    try:
        lemmatizer = _get_lemmatizer()
        pairs = lemmatizer.lemmatize(toks)  # [(token, lemma), ...]
        lemmas = [lemma for _, lemma in pairs]
        return lemmas
    except Exception:
        return toks
