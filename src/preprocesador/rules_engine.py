from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RuleHit:
    name: str
    label: str
    message: str
    boost: Optional[Dict[str, float]] = None


def load_rules(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _score_gte(scores: Dict[str, float], dim: str, value: float) -> bool:
    return float(scores.get(dim, 0.0)) >= float(value)


def _has_form(matched_forms: Dict[str, List[str]], dim: str, value: str) -> bool:
    return value in (matched_forms.get(dim) or [])

def load_rules_from_yaml(path: str | Path) -> List[Dict[str, Any]]:
    """
    Carga un archivo YAML de reglas y devuelve una lista de dicts (rules).
    Acepta dos formatos:
      - raíz con key 'rules': {rules: [ ... ]}
      - raíz como lista: [ ... ]
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))

    if data is None:
        return []

    if isinstance(data, dict) and "rules" in data:
        rules = data["rules"]
    else:
        rules = data

    if not isinstance(rules, list):
        raise ValueError(f"Formato inválido en {p}. Se esperaba lista de reglas o dict con 'rules'.")

    # Normaliza: cada regla debe ser dict
    out: List[Dict[str, Any]] = []
    for r in rules:
        if isinstance(r, dict):
            out.append(r)
        else:
            raise ValueError(f"Regla inválida (no es dict): {r}")
    return out

def apply_rules(
    rules_doc: Dict[str, Any],
    frame_name: str,
    scores: Dict[str, float],
    matched_forms: Dict[str, List[str]],
) -> List[RuleHit]:
    hits: List[RuleHit] = []
    rules = rules_doc.get("rules", []) or []

    def eval_clause(clause: Dict[str, Any]) -> bool:
        if "score_gte" in clause:
            d = clause["score_gte"]["dim"]
            v = clause["score_gte"]["value"]
            return _score_gte(scores, d, v)
        if "has_form" in clause:
            d = clause["has_form"]["dim"]
            v = clause["has_form"]["value"]
            return _has_form(matched_forms, d, v)
        return False

    for r in rules:
        rname = str(r.get("name", "unnamed_rule"))
        when = r.get("when", {}) or {}
        then = r.get("then", {}) or {}

        wf = when.get("frame")
        if wf and wf != frame_name:
            continue

        ok_all = True
        if "all" in when:
            ok_all = all(eval_clause(c) for c in (when["all"] or []))

        ok_any = True
        if "any" in when:
            ok_any = any(eval_clause(c) for c in (when["any"] or []))

        if ok_all and ok_any:
            boost = then.get("boost", None)
            if isinstance(boost, dict):
                boost = {str(k): float(v) for k, v in boost.items()}
            else:
                boost = None

            hits.append(
                RuleHit(
                    name=rname,
                    label=str(then.get("label", rname)),
                    message=str(then.get("message", "")),
                    boost=boost,
                )
            )

    return hits


def aggregate_boosts(dimensions: List[str], hits: List[RuleHit]) -> Dict[str, float]:
    boost = {d: 0.0 for d in dimensions}
    for h in hits:
        if h.boost:
            for d, v in h.boost.items():
                if d in boost:
                    boost[d] += float(v)
    return boost
