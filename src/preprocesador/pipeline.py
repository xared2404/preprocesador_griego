from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class PreprocessResult:
    frame: str
    scores: Dict[str, float]
    dominant_dimension: Optional[str]
    normalized_text: str
    tokens: List[str]
    lemmas: List[str]
    matched_forms: Dict[str, List[str]]
    matched_lemmas: Dict[str, List[str]]
    matches: Dict[str, List[Tuple[str, float]]]
    matches_lemmas: Dict[str, List[Tuple[str, float]]]
    note: str = ""


class CognitivePreprocessor:
    def __init__(self, frame):
        self.frame = frame

    def run(self, text: str) -> PreprocessResult:
        pack: Dict[str, Any] = self.frame.score(text)

        scores = pack["scores"]
        matched_forms = pack.get("matched_forms", {})
        matched_lemmas = pack.get("matched_lemmas", {})
        matches_forms = pack.get("matches_forms", {})
        matches_lemmas = pack.get("matches_lemmas", {})

        # evidencia principal para tie-break: forms + lemmas combinadas
        combined_matches: Dict[str, List[Tuple[str, float]]] = {}
        for d in self.frame.dimensions:
            combined_matches[d] = (matches_forms.get(d, []) or []) + (matches_lemmas.get(d, []) or [])

        # dominante determinista (no None)
        max_score = max(scores.values()) if scores else 0.0
        top = [d for d, s in scores.items() if s == max_score]

        if len(top) == 1:
            dominant = top[0]
        else:
            # tie-break por evidencia total
            def ev_count(dim: str) -> int:
                return len(combined_matches.get(dim, []) or [])

            top2 = sorted(top, key=lambda d: ev_count(d), reverse=True)
            best = top2[0]
            best_ev = ev_count(best)
            tied = [d for d in top2 if ev_count(d) == best_ev]

            if len(tied) == 1:
                dominant = best
            else:
                # tie-break final por orden del frame
                dominant = None
                for d in self.frame.dimensions:
                    if d in tied:
                        dominant = d
                        break
                if dominant is None:
                    dominant = tied[0]

        return PreprocessResult(
            frame=self.frame.name,
            scores=scores,
            dominant_dimension=dominant,
            normalized_text=pack["normalized_text"],
            tokens=pack.get("tokens", []),
            lemmas=pack.get("lemmas", []),
            matched_forms=matched_forms,
            matched_lemmas=matched_lemmas,
            matches=combined_matches,
            matches_lemmas=matches_lemmas,
            note="v3: evidencia por forma + evidencia por lema (CLTK). Ensemble listo.",
        )
