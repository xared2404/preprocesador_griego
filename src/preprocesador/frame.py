from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass(frozen=True)
class Frame:
    name: str
    description: str
    dimensions: List[str]
    lexicon: Dict[str, Dict[str, float]]

    def score(self, text: str) -> Dict[str, Any]:
        """
        v3 (robusto):
        - normaliza texto -> tokens
        - lematiza griego (best effort)
        - hace match comparando FORMAS NORMALIZADAS:
            normalize_text(term_lexicon) vs tokens_normalizados
          y para lemas:
            normalize_text(term_lexicon) vs lemmas_normalizados
        - evita doble conteo (si ya matcheó por forma, no lo suma por lema)
        """
        from .normalize import normalize_text, tokenize
        from .lemmatize import lemmatize_grc_best_effort

        normalized = normalize_text(text)
        tokens = tokenize(normalized)

        lemmas_raw = lemmatize_grc_best_effort(tokens)
        # Normalizamos lemas con la misma función para que acentos/diacríticos no rompan el match
        lemmas = [normalize_text(l) for l in lemmas_raw if l]

        scores: Dict[str, float] = {d: 0.0 for d in self.dimensions}

        # Evidencia: guardamos el término ORIGINAL del lexicón y su peso
        matches_forms: Dict[str, List[Tuple[str, float]]] = {d: [] for d in self.dimensions}
        matches_lemmas: Dict[str, List[Tuple[str, float]]] = {d: [] for d in self.dimensions}

        # Matched_*: guardamos la forma NORMALIZADA que efectivamente coincidió (útil para rules)
        matched_forms: Dict[str, List[str]] = {d: [] for d in self.dimensions}
        matched_lemmas: Dict[str, List[str]] = {d: [] for d in self.dimensions}

        tok_set = set(tokens)
        lemma_set = set(lemmas)

        # 1) match por FORMA normalizada
        for dim in self.dimensions:
            for term, w in (self.lexicon.get(dim, {}) or {}).items():
                term_norm = normalize_text(str(term))
                if term_norm in tok_set:
                    scores[dim] += float(w)
                    matches_forms[dim].append((str(term), float(w)))  # evidencia: original
                    matched_forms[dim].append(term_norm)             # para reglas: normalizado

        # 2) match por LEMA normalizado (sin doble conteo)
        for dim in self.dimensions:
            already = set(matched_forms[dim])  # normalizados ya contados por forma
            for term, w in (self.lexicon.get(dim, {}) or {}).items():
                term_norm = normalize_text(str(term))
                if term_norm in already:
                    continue
                if term_norm in lemma_set:
                    scores[dim] += float(w)
                    matches_lemmas[dim].append((str(term), float(w)))
                    matched_lemmas[dim].append(term_norm)

        # Ordena evidencia por peso
        for dim in self.dimensions:
            matches_forms[dim].sort(key=lambda x: x[1], reverse=True)
            matches_lemmas[dim].sort(key=lambda x: x[1], reverse=True)
        for d in self.dimensions:
              matched_forms[d] = sorted(set(matched_forms[d]))
              matched_lemmas[d] = sorted(set(matched_lemmas[d]))
        return {
            "normalized_text": normalized,
            "tokens": tokens,
            "lemmas": lemmas,  # ya normalizados
            "scores": scores,
            "matches_forms": matches_forms,
            "matches_lemmas": matches_lemmas,
            "matched_forms": matched_forms,
            "matched_lemmas": matched_lemmas,
        }
