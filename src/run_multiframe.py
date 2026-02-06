from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.preprocesador.pipeline import CognitivePreprocessor
from src.preprocesador.frame_loader import load_frame_from_yaml
from src.preprocesador.rules_engine import load_rules, apply_rules


def read_input(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def to_dict(res: Any) -> Dict[str, Any]:
    if isinstance(res, dict):
        return res
    if is_dataclass(res):
        return asdict(res)
    if hasattr(res, "__dict__"):
        return dict(res.__dict__)
    raise TypeError(f"Resultado no soportado: {type(res)}")


def flatten_for_csv(frame_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "frame": frame_name,
        "dominant_dimension": result.get("dominant_dimension"),
    }

    scores = result.get("scores", {}) or {}
    for dim, val in scores.items():
        row[f"score__{dim}"] = val

    matched_forms = result.get("matched_forms", {}) or {}
    for dim, forms in matched_forms.items():
        row[f"forms__{dim}"] = "|".join(forms) if forms else ""

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cognitive preprocessing with multiple YAML frames"
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default=None,
        help="Input text file (UTF-8). If omitted, stdin.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        required=True,
        help="List of YAML frame files",
    )
    parser.add_argument(
        "--rules",
        dest="rules_path",
        default=None,
        help="Rules YAML path",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Write full results to JSON",
    )
    parser.add_argument(
        "--csv",
        dest="csv_out",
        default=None,
        help="Write flattened results to CSV",
    )

    args = parser.parse_args()
    text = read_input(args.in_path)

    rules_doc = load_rules(args.rules_path) if args.rules_path else None

    results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    print("\n=== MULTI-FRAME COGNITIVE REPORT ===")

    for frame_path in args.frames:
        frame = load_frame_from_yaml(frame_path)
        pipe = CognitivePreprocessor(frame)
        res = to_dict(pipe.run(text))

        results.append(
            {
                "frame_path": str(frame_path),
                "frame_name": frame.name,
                "result": res,
            }
        )
        csv_rows.append(flatten_for_csv(frame.name, res))

        print(f"\nFRAME: {frame.name}")
        print("DOMINANT:", res.get("dominant_dimension"))
        print("SCORES:", res.get("scores"))

        # ---- RULES ----
        if rules_doc:
            hits = apply_rules(
                rules_doc,
                frame.name,
                res.get("scores", {}) or {},
                res.get("matched_forms", {}) or {},
            )
            if hits:
                print("\n--- RULE HITS ---")
                for h in hits:
                    print(f"{h.label}: {h.message}")

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['frame_name']}: {r['result'].get('dominant_dimension')}")

    # ---- JSON export ----
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[OK] JSON -> {p}")

    # ---- CSV export ----
    if args.csv_out:
        p = Path(args.csv_out)
        p.parent.mkdir(parents=True, exist_ok=True)

        all_keys = set()
        for row in csv_rows:
            all_keys |= set(row.keys())
        fieldnames = sorted(all_keys)

        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        print(f"[OK] CSV -> {p}")


if __name__ == "__main__":
    main()
