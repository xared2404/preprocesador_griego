from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def frame_semantic_scores(
    normalized_text: str,
    dimensions: List[str],
    lexicon: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Prototipo por dimensión = términos del lexicón (repetidos por peso con cap).
    Luego similarity(text, prototype_dim).
    """
    model = _get_model()

    prototypes: List[str] = []
    for d in dimensions:
        terms = lexicon.get(d, {}) or {}
        toks: List[str] = []
        for term, w in terms.items():
            k = int(min(max(round(float(w)), 1), 3))  # 1..3 repeticiones
            toks.extend([term] * k)
        prototypes.append(" ".join(toks) if toks else d)

    emb_text = model.encode([normalized_text], normalize_embeddings=False)[0]
    emb_dims = model.encode(prototypes, normalize_embeddings=False)

    out: Dict[str, float] = {}
    for d, e in zip(dimensions, emb_dims):
        out[d] = cosine(emb_text, e)

    return out
