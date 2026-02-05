from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import asdict, is_dataclass

from src.preprocesador.lexicon import attic_tragedy_frame
from src.preprocesador.pipeline import CognitivePreprocessor
from src.preprocesador.frame_loader import load_frame_from_yaml


def read_input(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def result_to_dict(res):
    """
    Normaliza el resultado del pipeline a dict.
    Soporta:
      - dict
      - dataclass (PreprocessResult)
      - objetos con __dict__
    """
    if isinstance(res, dict):
        return res
    if is_dataclass(res):
        return asdict(res)
    if hasattr(res, "__dict__"):
        return dict(res.__dict__)
    raise TypeError(f"Resultado no soportado: {type(res)}")


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive preprocessing pipeline (Greek tragedy)"
    )

    parser.add_argument(
        "--in",
        dest="in_path",
        default=None,
        help="Input text file (UTF-8). If omitted, read from stdin."
    )

    parser.add_argument(
        "--frame",
        dest="frame_path",
        default=None,
        help="Path to a YAML frame (default: built-in attic_tragedy_frame)"
    )

    args = parser.parse_args()

    text = read_input(args.in_path)

    if args.frame_path:
        frame = load_frame_from_yaml(args.frame_path)
    else:
        frame = attic_tragedy_frame

    pipe = CognitivePreprocessor(frame)
    res = pipe.run(text)
    result = result_to_dict(res)

    print("\n=== COGNITIVE PREPROCESSOR REPORT ===")
    print(f"FRAME: {frame.name}")
    print(f"DOMINANT: {result.get('dominant')}")
    print(f"SCORES: {result.get('scores')}")

    evidence = result.get("evidence") or {}
    if evidence:
        print("\n--- Evidence (matches) ---")
        for dim, ev in evidence.items():
            if ev:
                print(f"{dim}: {ev}")

    matched_forms = result.get("matched_forms") or {}
    if matched_forms:
        print("\n--- Matched forms (normalized) ---")
        for dim, forms in matched_forms.items():
            if forms:
                print(f"{dim}: {forms}")

    normalized_text = result.get("normalized_text")
    if normalized_text:
        print("\n--- Normalized text (preview) ---")
        print(normalized_text)

    lemma_count = result.get("lemma_count", None)
    if lemma_count is not None:
        print(f"\nLEMMA COUNT: {lemma_count}")

    note = result.get("note", "")
    if note:
        print("\n--- Note ---")
        print(note)


if __name__ == "__main__":
    main()
