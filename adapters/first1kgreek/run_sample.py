from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adapters.first1kgreek.extract_tei import extract_tei_text

from preprocesador.lexicon import attic_tragedy_frame
from preprocesador.pipeline import CognitivePreprocessor


def iter_xml_files(root: Path, only_grc1: bool = True) -> List[Path]:
    files = sorted(root.rglob("*.xml"))
    if only_grc1:
        files = [
            p for p in files
            if (".grc1." in p.name) or p.name.endswith(".grc1.xml") or ("opp-grc1" in p.name)
        ]
    return files


def parse_tlg_ids(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = path.parts
    tlg_author = None
    tlg_work = None

    for i, p in enumerate(parts):
        if p == "split" and i + 2 < len(parts):
            a = parts[i + 1]
            w = parts[i + 2]
            if a.startswith("tlg"):
                tlg_author = a
            if w.startswith("tlg"):
                tlg_work = w
            break

    layer = "grc1" if ("grc1" in path.name) else None
    return tlg_author, tlg_work, layer


def chunk_text(text: str, chunk_chars: int, overlap: int = 0) -> List[Tuple[int, int, str]]:
    if chunk_chars <= 0:
        return [(0, len(text), text)]
    if overlap < 0:
        overlap = 0
    step = max(1, chunk_chars - overlap)

    chunks: List[Tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + chunk_chars)
        chunks.append((i, j, text[i:j]))
        if j >= n:
            break
        i += step
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run CognitivePreprocessor over First1KGreek TEI/XML and emit JSONL (optionally chunked)."
    )
    ap.add_argument("--input", required=True, help="Directory containing TEI/XML files (e.g., data/First1KGreek/split).")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--limit", type=int, default=5, help="Max number of XML files to process (docs).")
    ap.add_argument("--min-chars", type=int, default=200, help="Skip docs with extracted Greek text shorter than this.")
    ap.add_argument("--only-grc1", action="store_true", help="Only process files with grc1 in filename.")
    ap.add_argument("--chunk-chars", type=int, default=4000, help="Chunk size in characters. Use 0 to disable.")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters.")
    ap.add_argument("--max-chunks-per-doc", type=int, default=10, help="Safety cap per document.")
    ap.add_argument("--text-preview-chars", type=int, default=280, help="Store a short preview of normalized_text (default 280). Use 0 to omit.")
    ap.add_argument("--include-debug-fields", action="store_true", help="Include tokens/lemmas/full normalized_text and match dicts (big).")
    args = ap.parse_args()

    in_dir = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = iter_xml_files(in_dir, only_grc1=bool(args.only_grc1))
    if not files:
        raise SystemExit(f"No matching XML files found under: {in_dir}")

    pre = CognitivePreprocessor(frame=attic_tragedy_frame)

    docs_done = 0
    scanned = len(files)

    with out_path.open("w", encoding="utf-8") as f:
        for fp in files:
            if docs_done >= args.limit:
                break

            tlg_author, tlg_work, layer = parse_tlg_ids(fp)
            ex = extract_tei_text(fp)

            if len(ex.text) < args.min_chars:
                continue

            chunks = chunk_text(ex.text, args.chunk_chars, overlap=args.chunk_overlap)
            chunks = chunks[: max(1, args.max_chunks_per_doc)]

            for chunk_id, (a, b, chunk) in enumerate(chunks):
                res = pre.run(chunk)

                row: Dict[str, object] = {
                    "source_path": ex.source_path,
                    "doc_id": ex.doc_id,
                    "urn": ex.urn,
                    "tlg_author": tlg_author,
                    "tlg_work": tlg_work,
                    "layer": layer,
                    "chunk_id": chunk_id,
                    "char_start": a,
                    "char_end": b,
                    "frame": res.frame,
                    "dominant_dimension": res.dominant_dimension,
                    "scores": res.scores,
                    "note": res.note,
                }

                # lightweight by default: include only a preview, or nothing
                if args.include_debug_fields:
                    row["normalized_text"] = res.normalized_text
                    row["tokens"] = res.tokens
                    row["lemmas"] = res.lemmas
                    row["matched_forms"] = res.matched_forms
                    row["matched_lemmas"] = res.matched_lemmas
                else:
                    if args.text_preview_chars and args.text_preview_chars > 0:
                        row["text_preview"] = res.normalized_text[: args.text_preview_chars]

                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            docs_done += 1

    print(f"[OK] wrote {out_path} docs={docs_done} scanned={scanned} (chunk_chars={args.chunk_chars})")


if __name__ == "__main__":
    main()
