from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class Frame:
    """
    Marco interpretativo explícito (frame cognitivo).

    - Integra normalización politónica
    - Tokenización
    - Lematización griega (CLTK, con fallback)
    - Scoring explicable con evidencia
    """
    name: str
    description: str = ""
    dimensions: List[str] = field(default_factory=list)
    lexicon: Dict[str, Dict[str, float]] = field(default_factory=dict)  # dim -> {term: weight}

    def score_with_matches(self, text: str) -> Dict[str, Any]:
        from .normalize import normalize_text, tokenize
        from .lemmatize import lemmatize_greek

        # --- preprocessing ---
        normalized = normalize_text(text)
        tokens = tokenize(normalized)
        lemmas = lemmatize_greek(normalized)

        tokset = set(tokens) | set(lemmas)

        # --- outputs ---
        scores: Dict[str, float] = {d: 0.0 for d in self.dimensions}
        matches: Dict[str, List[Tuple[str, float]]] = {d: [] for d in self.dimensions}
        matched_forms: Dict[str, List[str]] = {d: [] for d in self.dimensions}

        # --- scoring ---
        for dim in self.dimensions:
            terms = self.lexicon.get(dim, {})
            for term, w in terms.items():
                if not term:
                    continue

                term_norm = normalize_text(term)
                if term_norm in tokset:
                    scores[dim] += float(w)
                    matches[dim].append((term, float(w)))
                    matched_forms[dim].append(term_norm)

            # orden por peso
            matches[dim].sort(key=lambda x: x[1], reverse=True)
            matched_forms[dim] = sorted(set(matched_forms[dim]))

        return {
            "normalized_text": normalized,
            "tokens": tokens,
            "lemmas": lemmas,
            "scores": scores,
            "matches": matches,
            "matched_forms": matched_forms,
        }

    def score(self, text: str) -> Dict[str, float]:
        """Wrapper: conserva compatibilidad devolviendo solo scores."""
        return self.score_with_matches(text)["scores"]

    def top_terms(self, dim: str, k: int = 10) -> List[Tuple[str, float]]:
        terms = self.lexicon.get(dim, {})
        return sorted(terms.items(), key=lambda x: x[1], reverse=True)[:k]
