from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.preprocesador.lexicon import attic_tragedy_frame
from src.preprocesador.pipeline import CognitivePreprocessor


def read_text(path: Optional[str]) -> str:
    if path is None:
        # lee de stdin
        return sys.stdin.read()
    p = Path(path)
    return p.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Greek cognitive preprocessor pipeline on a text."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default=None,
        help="Input text file (UTF-8). If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Optional path to write full result as JSON.",
    )
    args = parser.parse_args()

    text = read_text(args.in_path)

    frame = attic_tragedy_frame()
    p = CognitivePreprocessor(frame)
    res = p.run(text)

    # Report humano
    print("\n=== COGNITIVE PREPROCESSOR REPORT ===")
    print("FRAME:", res.frame_name)
    print("DOMINANT:", res.dominant_dimension)
    print("SCORES:", res.scores)

    print("\n--- Evidence (matches) ---")
    for dim, evidence in res.matches.items():
        if not evidence:
            continue
        print(f"{dim}: {evidence}")

    print("\n--- Normalized text (preview) ---")
    preview = res.normalized_text[:400]
    print(preview + ("..." if len(res.normalized_text) > 400 else ""))

    print("\n--- Note ---")
    print(res.explanation.get("note", ""))

    # JSON export (opcional)
    if args.json_out:
        out = {
            "frame_name": res.frame_name,
            "dominant_dimension": res.dominant_dimension,
            "scores": res.scores,
            "matches": res.matches,
            "normalized_text": res.normalized_text,
            "tokens": res.tokens,
            "explanation": res.explanation,
        }
        Path(args.json_out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[OK] JSON escrito en: {args.json_out}")


if __name__ == "__main__":
    main()
