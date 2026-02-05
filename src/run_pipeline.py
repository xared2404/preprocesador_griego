from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.preprocesador.lexicon import attic_tragedy_frame
from src.preprocesador.pipeline import CognitivePreprocessor


def read_text(path: str | None) -> str:
    if path is None:
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cognitive preprocessing pipeline")
    parser.add_argument("--in", dest="in_path", default=None, help="Input text file (UTF-8). If omitted, read stdin.")
    args = parser.parse_args()

    text = read_text(args.in_path)

    frame = attic_tragedy_frame()
    pipeline = CognitivePreprocessor(frame)
    res = pipeline.run(text)

    print("\n=== COGNITIVE PREPROCESSOR REPORT ===")
    print("FRAME:", res.frame_name)
    print("DOMINANT:", res.dominant_dimension)
    print("SCORES:", res.scores)

    # Evidence (matches): soporta res.matches dict
    print("\n--- Evidence (matches) ---")
    matches = getattr(res, "matches", {}) or {}
    for dim, ev in matches.items():
        if ev:
            print(f"{dim}: {ev}")

    # Matched forms (normalized): si existe
    matched_forms = getattr(res, "matched_forms", {}) or {}
    if matched_forms:
        print("\n--- Matched forms (normalized) ---")
        for dim, forms in matched_forms.items():
            if forms:
                print(f"{dim}: {forms}")

    print("\n--- Normalized text (preview) ---")
    print(res.normalized_text[:400])

    # Lemma count: si existe
    lemmas = getattr(res, "lemmas", None)
    if lemmas is not None:
        print("\nLEMMA COUNT:", len(set(lemmas)))

    print("\n--- Note ---")
    note = ""
    if isinstance(res.explanation, dict):
        note = res.explanation.get("note", "")
    print(note)


if __name__ == "__main__":
    main()
