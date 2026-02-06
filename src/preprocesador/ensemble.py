from __future__ import annotations

from typing import Dict, List, Optional


def minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def combine_scores(
    dimensions: List[str],
    lexical_scores: Dict[str, float],
    semantic_scores: Optional[Dict[str, float]] = None,
    rule_boosts: Optional[Dict[str, float]] = None,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
) -> Dict[str, float]:
    """
    S_final(d) = alpha * norm(lex)(d) + beta * norm(sem)(d) + gamma * rule_boost(d)

    - norm(...) es min-max a [0,1]
    - rule_boost por default se interpreta ya en [0,1] (pero puedes usar valores pequeños).
    """
    semantic_scores = semantic_scores or {}
    rule_boosts = rule_boosts or {}

    lex_n = minmax_norm({d: float(lexical_scores.get(d, 0.0)) for d in dimensions})
    sem_n = minmax_norm({d: float(semantic_scores.get(d, 0.0)) for d in dimensions})

    out: Dict[str, float] = {}
    for d in dimensions:
        rb = float(rule_boosts.get(d, 0.0))
        out[d] = alpha * lex_n.get(d, 0.0) + beta * sem_n.get(d, 0.0) + gamma * rb
    return out


def choose_dominant(
    scores: Dict[str, float],
    evidence: Optional[Dict[str, list]] = None,
    dims_order: Optional[List[str]] = None,
) -> tuple[str | None, bool]:
    """
    Tie-break determinista:
      1) mayor score
      2) mayor cantidad de evidencia (len(evidence[dim]))
      3) primer dim según dims_order
    """
    if not scores:
        return None, False

    max_score = max(scores.values())
    top = [d for d, s in scores.items() if s == max_score]

    if len(top) == 1:
        return top[0], False

    evidence = evidence or {}
    dims_order = dims_order or list(scores.keys())

    def ev_count(d: str) -> int:
        return len(evidence.get(d, []) or [])

    # 2) evidencia
    top_sorted = sorted(top, key=lambda d: ev_count(d), reverse=True)
    best = top_sorted[0]
    best_ev = ev_count(best)
    tied = [d for d in top_sorted if ev_count(d) == best_ev]

    if len(tied) == 1:
        return best, True

    # 3) orden del frame
    for d in dims_order:
        if d in tied:
            return d, True

    return tied[0], True
