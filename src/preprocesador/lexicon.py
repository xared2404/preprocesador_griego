# src/preprocesador/lexicon.py

from __future__ import annotations

from .frame import Frame

def attic_tragedy_frame() -> Frame:
    """
    Frame inicial (muy baseline) inspirado en tragedia ática.
    Ojo: por ahora usa string matching. Luego lo elevamos a griego politónico + lematización.
    """
    dims = ["nomos", "dike", "hybris", "ritual", "polis"]

    lex = {
        "nomos": {
            "νόμος": 2.0,
            "nomos": 1.0,
            "law": 0.8,
            "ley": 0.8,
        },
        "dike": {
            "δίκη": 2.0,
            "dike": 1.0,
            "justice": 0.8,
            "justicia": 0.8,
            "punishment": 0.6,
            "castigo": 0.6,
        },
        "hybris": {
            "ὕβρις": 2.0,
            "hybris": 1.0,
            "excess": 0.7,
            "exceso": 0.7,
            "pride": 0.6,
            "soberbia": 0.6,
        },
        "ritual": {
            "θυσία": 1.8,
            "sacrifice": 0.7,
            "sacrificio": 0.7,
            "altar": 0.6,
            "ritual": 0.6,
        },
        "polis": {
            "πόλις": 1.8,
            "polis": 1.0,
            "ciudad": 0.6,
            "citizen": 0.6,
            "ciudadano": 0.6,
            "tyrant": 0.6,
            "tirano": 0.6,
        },
    }

    return Frame(
        name="attic_tragedy_v0",
        description="Frame baseline para tragedia ática (nomos/dike/hybris/ritual/polis).",
        dimensions=dims,
        lexicon=lex,
    )
