from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.preprocesador.frame_loader import load_frame_from_yaml
from src.preprocesador.pipeline import CognitivePreprocessor
from src.preprocesador.rules_engine import load_rules, apply_rules, aggregate_boosts
from src.preprocesador.embeddings import frame_semantic_scores
from src.preprocesador.ensemble import combine_scores, choose_dominant


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


def write_json(path: str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    fieldnames = sorted(keys)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Final pipeline: lexical + rules + semantic ensemble")
    p.add_argument("--in", dest="in_path", default=None)
    p.add_argument("--frame", dest="frame_path", required=True)
    p.add_argument("--rules", dest="rules_path", default=None)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--json", dest="json_out", default=None, help="Write full report to JSON")
    p.add_argument("--csv", dest="csv_out", default=None, help="Write flattened report to CSV (1 row)")
    args = p.parse_args()

    text = read_input(args.in_path)

    frame = load_frame_from_yaml(args.frame_path)
    pipe = CognitivePreprocessor(frame)
    res = to_dict(pipe.run(text))

    lexical_scores = res.get("scores", {}) or {}
    evidence = res.get("matches", {}) or {}
    matched_forms = res.get("matched_forms", {}) or {}
    matched_lemmas = res.get("matched_lemmas", {}) or {}

    # Rules
    hits = []
    rule_boosts = {d: 0.0 for d in frame.dimensions}
    if args.rules_path:
        rules_doc = load_rules(args.rules_path)
        hits = apply_rules(rules_doc, frame.name, lexical_scores, matched_forms)
        rule_boosts = aggregate_boosts(frame.dimensions, hits)

    # Semantic
    semantic_scores = frame_semantic_scores(
        normalized_text=res["normalized_text"],
        dimensions=frame.dimensions,
        lexicon=frame.lexicon,
    )

    # Ensemble
    final_scores = combine_scores(
        dimensions=frame.dimensions,
        lexical_scores=lexical_scores,
        semantic_scores=semantic_scores,
        rule_boosts=rule_boosts,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    dominant_final, tie_final = choose_dominant(final_scores, evidence=evidence, dims_order=frame.dimensions)

    report = {
        "frame": frame.name,
        "input_path": args.in_path,
        "normalized_text": res.get("normalized_text"),
        "tokens": res.get("tokens"),
        "lemmas": res.get("lemmas"),
        "lexical_scores": lexical_scores,
        "semantic_scores": semantic_scores,
        "rule_hits": [
            {"label": h.label, "message": h.message, "boost": h.boost} for h in hits
        ],
        "rule_boosts": rule_boosts,
        "final_scores": final_scores,
        "dominant_lexical": res.get("dominant_dimension"),
        "dominant_final": dominant_final,
        "tie_break_final": tie_final,
    }

    print("\n=== FINAL ENSEMBLE REPORT ===")
    print("FRAME:", frame.name)

    print("\nDOMINANT (lexical):", report["dominant_lexical"])
    print("SCORES (lexical):", lexical_scores)

    print("\n--- Matched forms (normalized) ---")
    for d, forms in matched_forms.items():
        if forms:
            print(f"{d}: {forms}")

    print("\n--- Matched lemmas (grc) ---")
    for d, lems in matched_lemmas.items():
        if lems:
            print(f"{d}: {lems}")

    print("\nSCORES (semantic cosine):", semantic_scores)

    if args.rules_path:
        print("\n--- RULE HITS ---")
        if hits:
            for h in hits:
                print(f"{h.label}: {h.message} | boost={h.boost}")
        else:
            print("(none)")
        print("RULE BOOSTS:", rule_boosts)

    print("\nSCORES (FINAL ensemble):", final_scores)
    print("DOMINANT (FINAL):", dominant_final, "| tie_break:", tie_final)

    # Exports
    if args.json_out:
        write_json(args.json_out, report)
        print(f"\n[OK] JSON -> {args.json_out}")

    if args.csv_out:
        row = {"frame": frame.name, "dominant_final": dominant_final, "tie_break_final": tie_final}
        for d in frame.dimensions:
            row[f"lex__{d}"] = float(lexical_scores.get(d, 0.0))
            row[f"sem__{d}"] = float(semantic_scores.get(d, 0.0))
            row[f"final__{d}"] = float(final_scores.get(d, 0.0))
            row[f"boost__{d}"] = float(rule_boosts.get(d, 0.0))
            row[f"forms__{d}"] = "|".join(matched_forms.get(d, []) or [])
            row[f"lemmas__{d}"] = "|".join(matched_lemmas.get(d, []) or [])
        write_csv(args.csv_out, [row])
        print(f"[OK] CSV -> {args.csv_out}")


if __name__ == "__main__":
    main()
