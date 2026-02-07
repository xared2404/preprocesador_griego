from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

FrameLike = Any
FrameFactory = Callable[[], FrameLike]


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
    def __init__(self, frame: Union[FrameLike, FrameFactory]):
        # Accept either a Frame object with .score(), or a factory function returning it.
        self.frame = frame() if callable(frame) else frame

    def run(self, text: str) -> PreprocessResult:
        pack: Dict[str, Any] = self.frame.score(text)

        scores: Dict[str, float] = pack["scores"]
        dominant_dimension: Optional[str] = pack.get("dominant_dimension")

        normalized_text: str = pack.get("normalized_text", "")
        tokens: List[str] = pack.get("tokens", [])
        lemmas: List[str] = pack.get("lemmas", [])

        matched_forms: Dict[str, List[str]] = pack.get("matched_forms", {})
        matched_lemmas: Dict[str, List[str]] = pack.get("matched_lemmas", {})

        # Keep compatibility with older/newer naming
        matches: Dict[str, List[Tuple[str, float]]] = (
            pack.get("matches")
            or pack.get("matches_forms")
            or {}
        )
        matches_lemmas: Dict[str, List[Tuple[str, float]]] = (
            pack.get("matches_lemmas")
            or {}
        )

        note: str = pack.get("note", "")

        # Frame id (string) should be stable for reproducibility
        frame_id = pack.get("frame", getattr(self.frame, "name", None)) or "unknown_frame"

        return PreprocessResult(
            frame=str(frame_id),
            scores=scores,
            dominant_dimension=dominant_dimension,
            normalized_text=normalized_text,
            tokens=tokens,
            lemmas=lemmas,
            matched_forms=matched_forms,
            matched_lemmas=matched_lemmas,
            matches=matches,
            matches_lemmas=matches_lemmas,
            note=note,
        )
