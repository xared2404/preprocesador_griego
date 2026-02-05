from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import asdict, is_dataclass

from src.preprocesador.pipeline import CognitivePreprocessor
from src.preprocesador.frame_loader import load_frame_from_yaml


def read_input(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def to_dict(res):
    if isinstance(res, dict):
        return res
    if is_dataclass(res):
        return asdict(res)
    return dict(res.__dict__)


def main():
    parser = argparse.ArgumentParser(
        description="Run cognitive preprocessing with multiple frames"
    )
    parser.add_argument("--in", dest="in_path", default=None)
    parser.add_argument(
        "--frames",
        nargs="+",
        required=True,
        help="List of YAML frame files"
    )

    args = parser.parse_args()

    text = read_input(args.in_path)

    print("\n=== MULTI-FRAME COGNITIVE REPORT ===")

    results = []

    for frame_path in args.frames:
        frame = load_frame_from_yaml(frame_path)
        pipe = CognitivePreprocessor(frame)
        res = to_dict(pipe.run(text))

        results.append({
            "frame": frame.name,
            "dominant": res.get("dominant_dimension"),
            "scores": res.get("scores"),
            "matched_forms": res.get("matched_forms", {}),
        })

        print(f"\nFRAME: {frame.name}")
        print("DOMINANT:", res.get("dominant_dimension"))
        print("SCORES:", res.get("scores"))

    # resumen comparativo
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['frame']}: {r['dominant']}")

if __name__ == "__main__":
    main()
