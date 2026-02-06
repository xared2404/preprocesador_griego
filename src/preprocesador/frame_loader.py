from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from .frame import Frame


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_frame_from_yaml(path: str | Path) -> Frame:
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))

    _require(isinstance(doc, dict), f"YAML inválido: raíz debe ser dict ({p})")

    name = doc.get("name")
    description = doc.get("description", "")
    dimensions = doc.get("dimensions")
    lexicon = doc.get("lexicon")

    _require(isinstance(name, str) and name.strip(), f"YAML sin 'name' válido ({p})")
    _require(isinstance(dimensions, list) and all(isinstance(x, str) for x in dimensions),
             f"YAML 'dimensions' debe ser lista[str] ({p})")
    _require(isinstance(lexicon, dict),
             f"YAML 'lexicon' debe ser dict (ej. dim -> term->weight) ({p})")

    # Normaliza: asegura que cada dim tenga dict term->float
    norm_lex: Dict[str, Dict[str, float]] = {}
    for d in dimensions:
        block = lexicon.get(d, {}) or {}
        _require(isinstance(block, dict), f"lexicon['{d}'] debe ser dict ({p})")
        norm_lex[d] = {str(k): float(v) for k, v in block.items()}

    return Frame(
        name=name,
        description=str(description),
        dimensions=list(dimensions),
        lexicon=norm_lex,
    )
