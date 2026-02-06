from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

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


def main() -> None:
    p = argparse.ArgumentParser(description="Final pipeline: lexical + rules + semantic ensemble")
    p.add_argument("--in", dest="in_path", default=None)
    p.add_argument("--frame", dest="frame_path", required=True)
    p.add_argument("--rules", dest="rules_path", default=None)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.1)
    args = p.parse_args()

    text = read_input(args.in_path)

    frame = load_frame_from_yaml(args.frame_path)
    pipe = CognitivePreprocessor(frame)
    res = to_dict(pipe.run(text))

    lexical_scores = res.get("scores", {}) or {}
    evidence = res.get("matches", {}) or {}          # evidencia por dim (si existe)
    matched_forms = res.get("matched_forms", {}) or {}

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

    # Print report
    print("\n=== FINAL ENSEMBLE REPORT ===")
    print("FRAME:", frame.name)

    dom_lex = res.get("dominant_dimension")
    print("\nDOMINANT (lexical):", dom_lex)
    print("SCORES (lexical):", lexical_scores)

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


if __name__ == "__main__":
    main()
