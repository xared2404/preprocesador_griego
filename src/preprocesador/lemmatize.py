from __future__ import annotations

from typing import List


def lemmatize_grc_best_effort(tokens: List[str]) -> List[str]:
    """
    CLTK 1.5 puede exponer lemas de distintas formas según pipeline/modelo.
    Esta función intenta varias rutas y nunca rompe: si no encuentra, devuelve [].
    """
    try:
        from cltk.nlp import NLP  # type: ignore

        nlp = NLP(language="grc", suppress_banner=True)
        doc = nlp.analyze(" ".join(tokens))

        lemmas: List[str] = []

        # Ruta A: CLTK words con atributo .lemma
        for w in getattr(doc, "words", []) or []:
            lemma = getattr(w, "lemma", None)
            if lemma:
                lemmas.append(str(lemma))

        if lemmas:
            return lemmas

        # Ruta B: tokens estilo spaCy: .lemma_ (string)
        for t in getattr(doc, "tokens", []) or []:
            lemma_ = getattr(t, "lemma_", None)
            if lemma_:
                lemmas.append(str(lemma_))

        if lemmas:
            return lemmas

        # Ruta C: si CLTK guardó el spaCy doc
        spacy_doc = getattr(doc, "spacy_doc", None)
        if spacy_doc is not None:
            for t in spacy_doc:
                lemma_ = getattr(t, "lemma_", None)
                if lemma_:
                    lemmas.append(str(lemma_))
            if lemmas:
                return lemmas

        return []
    except Exception:
        return []


def lemmatize_grc(tokens: List[str]) -> List[str]:
    return lemmatize_grc_best_effort(tokens)
