# src/run_batch_final.py
"""
Batch runner (FINAL ensemble) for many input .txt files.

Usage:
  ./.venv/bin/python -m src.run_batch_final \
    --in_dir data/corpus_batch \
    --frame frames/attic_tragedy.yaml \
    --rules rules/rules.yaml \
    --out_json reports/batch_final.json \
    --out_csv reports/batch_final.csv

What it does (best-effort):
- Loads a Frame from YAML
- Loads rules from YAML (optional)
- For each .txt in --in_dir:
    - frame.score(text)  -> lexical pack (scores + matched forms/lemmas + normalized text if provided)
    - semantic scores via sentence-transformers (if available)
    - apply_rules(...) (if available)
    - combine into an ensemble "final" score
- Writes:
    - JSON: list of per-file dicts
    - CSV: one row per file (flattened columns)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---- Imports from your package (keep them minimal and robust) ----
from src.preprocesador.frame_loader import load_frame_from_yaml

# rules_engine exists in your repo (you already tested imports)
from src.preprocesador.rules_engine import apply_rules, load_rules_from_yaml

# sentence-transformers semantic (optional / may fail if not installed in current interpreter)
try:
    from src.preprocesador.embeddings import frame_semantic_scores  # type: ignore
except Exception:
    frame_semantic_scores = None  # type: ignore


# ----------------------------- helpers -----------------------------

def _ensure_parent_dir(path_str: Optional[str]) -> None:
    if not path_str:
        return
    p = Path(path_str)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass-like or dict-like objects to dict safely."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    # last resort: try __dict__
    d = getattr(obj, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {}


def _dominant_label(scores: Dict[str, float]) -> Tuple[Optional[str], bool]:
    """Return (dominant_label, tie_break). tie_break True if tie for max."""
    if not scores:
        return None, False
    # stable sort by score desc then label
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_label, best_score = items[0]
    # check tie
    tied = [lbl for lbl, sc in items if sc == best_score]
    tie_break = len(tied) > 1
    return best_label, tie_break


def _coerce_float_dict(d: Any, labels: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {k: 0.0 for k in labels}
    if isinstance(d, dict):
        for k in labels:
            v = d.get(k, 0.0)
            try:
                out[k] = float(v)
            except Exception:
                out[k] = 0.0
    return out


def _flatten_list(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        # keep deterministic order for sets
        if isinstance(v, set):
            v = sorted(v)
        return "|".join(str(x) for x in v)
    return str(v)


def _flatten_forms_map(forms_map: Any, labels: List[str]) -> Dict[str, str]:
    """
    forms_map expected: {label: [forms...]} OR {label: set(...)} OR {label: str}
    Output: {label: "a|b|c"}
    """
    out: Dict[str, str] = {k: "" for k in labels}
    if isinstance(forms_map, dict):
        for k in labels:
            out[k] = _flatten_list(forms_map.get(k))
    return out


def _ensemble_scores(
    labels: List[str],
    lex_scores: Dict[str, float],
    sem_scores: Dict[str, float],
    boosts: Dict[str, float],
    w_lex: float = 0.35,
    w_sem: float = 0.60,
    w_rule: float = 1.00,
) -> Dict[str, float]:
    """
    Simple ensemble:
      final = w_lex * lex + w_sem * sem + w_rule * boost
    (weights chosen to roughly match your earlier “final ensemble” feel)
    """
    out: Dict[str, float] = {}
    for k in labels:
        out[k] = (w_lex * float(lex_scores.get(k, 0.0))) + (w_sem * float(sem_scores.get(k, 0.0))) + (w_rule * float(boosts.get(k, 0.0)))
    return out


def _rules_boosts_from_hits(labels: List[str], hits: Any) -> Dict[str, float]:
    """
    hits expected from apply_rules: list of dicts or dataclass-like objects,
    each with 'boost' mapping.
    """
    boosts: Dict[str, float] = {k: 0.0 for k in labels}
    if not hits:
        return boosts
    if not isinstance(hits, (list, tuple)):
        hits = [hits]
    for h in hits:
        hd = _safe_dict(h)
        b = hd.get("boost") or hd.get("boosts") or {}
        if isinstance(b, dict):
            for k in labels:
                try:
                    boosts[k] += float(b.get(k, 0.0))
                except Exception:
                    pass
    return boosts


def _pretty_rule_hits(hits: Any) -> List[Dict[str, Any]]:
    """Keep rule hits JSON-friendly."""
    if not hits:
        return []
    out: List[Dict[str, Any]] = []
    if not isinstance(hits, (list, tuple)):
        hits = [hits]
    for h in hits:
        out.append(_safe_dict(h))
    return out


def _try_semantic_scores(text_for_sem: str, frame_obj: Any, labels: List[str]) -> Dict[str, float]:
    """
    Calls your src.preprocesador.embeddings.frame_semantic_scores if present.
    Because we don't know the exact signature in every iteration, we try
    a few common call patterns.
    """
    if frame_semantic_scores is None:
        return {k: 0.0 for k in labels}

    # Try common signatures:
    # 1) frame_semantic_scores(text, frame)
    # 2) frame_semantic_scores(frame, text)
    # 3) frame_semantic_scores(text, frame, labels)
    # 4) frame_semantic_scores(frame, text, labels)
    for args in [
        (text_for_sem, frame_obj),
        (frame_obj, text_for_sem),
        (text_for_sem, frame_obj, labels),
        (frame_obj, text_for_sem, labels),
    ]:
        try:
            res = frame_semantic_scores(*args)  # type: ignore
            if isinstance(res, dict):
                return _coerce_float_dict(res, labels)
        except Exception:
            continue

    # If nothing worked, just return zeros (do not crash batch)
    return {k: 0.0 for k in labels}


# ----------------------------- main logic -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch FINAL ensemble runner.")
    p.add_argument("--in_dir", required=True, help="Directory with input .txt files (UTF-8).")
    p.add_argument("--frame", required=True, help="Path to frame YAML (e.g., frames/attic_tragedy.yaml).")
    p.add_argument("--rules", default=None, help="Path to rules YAML (optional).")
    p.add_argument("--out_json", default=None, help="Output JSON path (optional).")
    p.add_argument("--out_csv", default=None, help="Output CSV path (optional).")
    p.add_argument("--glob", default="*.txt", help="Glob pattern inside in_dir (default: *.txt).")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"[ERROR] --in_dir no existe o no es directorio: {in_dir}")

    frame = load_frame_from_yaml(args.frame)

    # labels come from frame (keys in lexicon / dims). We infer from frame.score output if needed.
    # Best effort: try frame.labels, frame.concepts, frame.lexicon keys, etc.
    labels: List[str] = []
    for attr in ("labels", "concepts", "dimensions", "dims"):
        v = getattr(frame, attr, None)
        if isinstance(v, (list, tuple)) and v:
            labels = [str(x) for x in v]
            break
    if not labels:
        lexicon = getattr(frame, "lexicon", None)
        if isinstance(lexicon, dict) and lexicon:
            labels = list(lexicon.keys())

    rules = None
    if args.rules:
        rules = load_rules_from_yaml(args.rules)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"[ERROR] No encontré archivos con patrón '{args.glob}' en {in_dir}")

    results: List[Dict[str, Any]] = []

    for fp in files:
        text = _read_text(fp)

        # lexical pack
        pack = frame.score(text)  # IMPORTANT: Frame has score(), not lexical_scores()
        pack = _safe_dict(pack)

        lex_scores_raw = pack.get("scores", {}) or {}
        lex_scores = _coerce_float_dict(lex_scores_raw, labels or list(lex_scores_raw.keys()))

        # If labels were unknown, set them now from this first file
        if not labels:
            labels = sorted(lex_scores.keys())

        normalized_text = pack.get("normalized_text") or pack.get("normalized") or text
        matched_forms = pack.get("matched_forms") or {}
        matched_lemmas = pack.get("matched_lemmas") or pack.get("lemmas_matched") or {}

        # semantic (best-effort)
        sem_scores = _try_semantic_scores(str(normalized_text), frame, labels)

        # rules (best-effort)
        hits = []
        if rules is not None:
            try:
                # Try signatures:
                # apply_rules(text, scores, rules)
                # apply_rules(text, scores, rules, frame=frame)
                # apply_rules(text, scores, rules, labels=labels)
                for call in [
                    (str(normalized_text), lex_scores, rules),
                    (str(normalized_text), lex_scores, rules, {"frame": frame}),
                    (str(normalized_text), lex_scores, rules, {"labels": labels}),
                    (str(normalized_text), lex_scores, rules, {"frame": frame, "labels": labels}),
                ]:
                    try:
                        if len(call) == 3:
                            hits = apply_rules(call[0], call[1], call[2])  # type: ignore
                        else:
                            hits = apply_rules(call[0], call[1], call[2], **call[3])  # type: ignore
                        break
                    except Exception:
                        continue
            except Exception:
                hits = []

        boosts = _rules_boosts_from_hits(labels, hits)

        # final ensemble
        final_scores = _ensemble_scores(labels, lex_scores, sem_scores, boosts)
        dominant_final, tie_break_final = _dominant_label(final_scores)

        record: Dict[str, Any] = {
            "frame": getattr(frame, "name", None) or pack.get("frame") or Path(args.frame).stem,
            "input_path": str(fp),
            "normalized_text": normalized_text,
            "tokens": pack.get("tokens") or [],
            "lemmas": pack.get("lemmas") or [],
            "lex_scores": lex_scores,
            "sem_scores": sem_scores,
            "rule_hits": _pretty_rule_hits(hits),
            "rule_boosts": boosts,
            "final_scores": final_scores,
            "dominant_final": dominant_final,
            "tie_break_final": tie_break_final,
            "matched_forms": matched_forms,
            "matched_lemmas": matched_lemmas,
        }
        results.append(record)

        print(f"[OK] {fp.name} -> dominant_final={dominant_final}")

    # outputs
    if args.out_json:
        _ensure_parent_dir(args.out_json)
        Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] JSON -> {args.out_json}")

    if args.out_csv:
        _ensure_parent_dir(args.out_csv)

        # Build a flat row per file, similar to your final_antigone.csv style
        fieldnames: List[str] = []
        # fixed columns
        fixed_cols = ["frame", "input_path", "dominant_final", "tie_break_final"]
        fieldnames.extend(fixed_cols)

        # per-label columns (lex/sem/final/boost/forms/lemmas)
        for prefix in ["lex__", "sem__", "final__", "boost__", "forms__", "lemmas__"]:
            for k in labels:
                fieldnames.append(f"{prefix}{k}")

        with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for r in results:
                row: Dict[str, Any] = {k: "" for k in fieldnames}
                row["frame"] = r.get("frame", "")
                row["input_path"] = r.get("input_path", "")
                row["dominant_final"] = r.get("dominant_final", "")
                row["tie_break_final"] = r.get("tie_break_final", False)

                lex = r.get("lex_scores", {}) or {}
                sem = r.get("sem_scores", {}) or {}
                fin = r.get("final_scores", {}) or {}
                boo = r.get("rule_boosts", {}) or {}
                forms_map = r.get("matched_forms", {}) or {}
                lemmas_map = r.get("matched_lemmas", {}) or {}

                forms_flat = _flatten_forms_map(forms_map, labels)
                lemmas_flat = _flatten_forms_map(lemmas_map, labels)

                for k in labels:
                    row[f"lex__{k}"] = lex.get(k, 0.0)
                    row[f"sem__{k}"] = sem.get(k, 0.0)
                    row[f"final__{k}"] = fin.get(k, 0.0)
                    row[f"boost__{k}"] = boo.get(k, 0.0)
                    row[f"forms__{k}"] = forms_flat.get(k, "")
                    row[f"lemmas__{k}"] = lemmas_flat.get(k, "")

                w.writerow(row)

        print(f"[OK] CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
