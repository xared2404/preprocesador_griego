from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from .frame import Frame


@dataclass
class PreprocessResult:
    lemmas: List[str]
    matched_forms: Dict[str, List[str]]
    frame_name: str
    normalized_text: str
    tokens: List[str]
    scores: Dict[str, float]
    matches: Dict[str, List[Tuple[str, float]]]
    dominant_dimension: Optional[str]
    explanation: Dict[str, Any]


class CognitivePreprocessor:
    def __init__(self, frame: Frame):
        self.frame = frame

    def run(self, text: str) -> PreprocessResult:
        out = self.frame.score_with_matches(text)
        scores = out["scores"]

        dominant = None
        if scores:
            dominant = max(scores.items(), key=lambda x: x[1])[0]

        explanation = {
            "frame_description": self.frame.description,
            "why_dominant": {
                "dimension": dominant,
                "evidence": out["matches"].get(dominant, []) if dominant else [],
            },
                "note": "v2: normalización politónica + evidencia léxica + lematización griega (CLTK). Próximo: embeddings y/o reglas más finas.",        }

        return PreprocessResult(
            frame_name=self.frame.name,
            normalized_text=out["normalized_text"],
            tokens=out["tokens"],
            scores=scores,
            matches=out["matches"],
            lemmas=out["lemmas"],
            matched_forms=out["matched_forms"],
            dominant_dimension=dominant,
            explanation=explanation,
        )
