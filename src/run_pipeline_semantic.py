from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from src.preprocesador.frame_loader import load_frame_from_yaml
from src.preprocesador.pipeline import CognitivePreprocessor
from src.preprocesador.embeddings import frame_semantic_scores


def read_input(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def to_dict(res: Any) -> Dict[str, Any]:
    if isinstance(res, dict):
        return res
    if is_dataclass(res):
        return asdict(res)
    return dict(res.__dict__)


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline + semantic scoring (post-frame embeddings)")
    p.add_argument("--in", dest="in_path", default=None)
    p.add_argument("--frame", dest="frame_path", required=True)
    args = p.parse_args()

    text = read_input(args.in_path)
    frame = load_frame_from_yaml(args.frame_path)

    pipe = CognitivePreprocessor(frame)
    res = to_dict(pipe.run(text))

    sem = frame_semantic_scores(
        normalized_text=res["normalized_text"],
        dimensions=frame.dimensions,
        lexicon=frame.lexicon,
    )

    print("\n=== PIPELINE + SEMANTIC REPORT ===")
    print("FRAME:", frame.name)
    print("DOMINANT (lexical):", res.get("dominant_dimension"))
    print("SCORES (lexical):", res.get("scores"))
    print("\nSCORES (semantic cosine):", sem)


if __name__ == "__main__":
    main()
