from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import re

try:
    from lxml import etree  # type: ignore
    _HAVE_LXML = True
except Exception:
    import xml.etree.ElementTree as etree  # type: ignore
    _HAVE_LXML = False


@dataclass
class ExtractedTEI:
    source_path: str
    doc_id: str
    urn: Optional[str]
    text: str


_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_SEG = re.compile(r"(?:\n+|(?<=[\.\!\?\;\··])\s+)")

# Expanded: also remove common critical apparatus / Latin editorial markers
_BOILER = re.compile(
    r"(text encoded|encoded in accordance|epidoc|creative commons|"
    r"archive\.org|project manager|technical advisor|university|"
    r"european social fund|digital divide|data corrected|available under|"
    r"argumentum|codicibus|foedatum|correxi|secundum|addidi|predicentes|"
    r"\bom\.\b|\btextus\b|\bscr\.\b|\blm:\b|"
    r"\bg\b|\bl\b|\bb\b)",
    re.IGNORECASE,
)


def _collapse_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _is_greek_char(ch: str) -> bool:
    o = ord(ch)
    return (0x0370 <= o <= 0x03FF) or (0x1F00 <= o <= 0x1FFF)


def _greek_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    g = sum(1 for c in letters if _is_greek_char(c))
    return g / len(letters)


def filter_mostly_greek(text: str, min_ratio: float = 0.35) -> str:
    text = _URL.sub(" ", text)

    segments = [seg.strip() for seg in _SEG.split(text) if seg.strip()]
    keep: List[str] = []
    for seg in segments:
        if _BOILER.search(seg):
            continue
        r = _greek_ratio(seg)
        if r >= min_ratio:
            keep.append(seg)

    return _collapse_ws(" ".join(keep))


def extract_tei_text(path: Path) -> ExtractedTEI:
    raw = path.read_bytes()

    urn = None
    doc_id = path.stem

    if _HAVE_LXML:
        root = etree.fromstring(raw)
        nsmap = root.nsmap or {}

        def xpath(expr: str):
            return root.xpath(expr, namespaces=nsmap)

        candidates = xpath("//*[local-name()='idno']/text()")
        if candidates:
            urn = _collapse_ws(str(candidates[0]))
        if urn is None:
            xmlid = root.get("{http://www.w3.org/XML/1998/namespace}id")
            if xmlid:
                urn = xmlid

        body_nodes = xpath("//*[local-name()='text']//*[local-name()='body']")
        if body_nodes:
            raw_text = "".join(body_nodes[0].itertext())
        else:
            text_nodes = xpath("//*[local-name()='text']")
            if text_nodes:
                raw_text = "".join(text_nodes[0].itertext())
            else:
                raw_text = "".join(root.itertext())
    else:
        root = etree.fromstring(raw)
        raw_text = "".join(root.itertext())

    raw_text = _collapse_ws(raw_text)
    clean = filter_mostly_greek(raw_text, min_ratio=0.35)

    return ExtractedTEI(
        source_path=str(path),
        doc_id=doc_id,
        urn=urn,
        text=clean,
    )
