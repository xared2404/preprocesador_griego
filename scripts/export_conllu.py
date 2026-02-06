from __future__ import annotations

import json
import re
from pathlib import Path

TOKENS = Path("outputs/parsed_tokens.jsonl")
OUT = Path("outputs/parsed.conllu")

# split sentences by token record that is punctuation "." or by blank line logic
# Here we rebuild sentences by scanning tokens and splitting on sentence-final punctuation.
SENT_FINAL = {".", "·", ";", ";", "?", "!"}

def safe(s: str | None) -> str:
    if not s:
        return "_"
    s = str(s).strip()
    return s if s else "_"

def clean_feats(feats: str) -> str:
    """
    FEATS in CoNLL-U must be '_' or 'Key=Val|Key=Val'.
    Also guard against accidental TokKey concatenation or truncations.
    """
    feats = (feats or "").strip()
    # --- FEATS_SANITIZER_V3 ---
    # Arregla casos como: "Case=NoTokKey=συ" donde TokKey se pegó al valor de Case
    if isinstance(feats, str) and feats not in ("", "_"):
        # 1) Si TokKey quedó pegado en el valor de Case (sin separador)
        feats = re.sub(r"(Case=[A-Za-z]+)TokKey=[^|\s]+", r"\1", feats)
        # 2) Normaliza Case=No* y Case=Nomm -> Case=Nom
        feats = re.sub(r"Case=No[^|\s]*", "Case=Nom", feats)
        feats = re.sub(r"Case=Nom+m", "Case=Nom", feats)
        # 3) Si quedó TokKey suelto en feats, bórralo
        feats = re.sub(r"\bTokKey=[^|\s]+\b", "", feats)
        # 4) Limpieza de separadores
        feats = re.sub(r"\|{2,}", "|", feats).strip("|")
        if feats == "":
            feats = "_"
    # --- /FEATS_SANITIZER_V3 ---

    if not feats:
        return "_"

    # If someone accidentally concatenated TokKey into feats, cut it away
    feats = re.sub(r"TokKey=.*$", "", feats).strip()

    # Common truncation fix (robusto):
    # - Case=No / Case=Nom / Case=Nomm / Case=NoTokKey... => Case=Nom
    feats = re.sub(r"Case=No[a-zA-Z]*", "Case=Nom", feats)
    feats = re.sub(r"Case=Nom+m", "Case=Nom", feats)

    # Remove trailing pipes
    feats = feats.strip("|")

    return feats if feats else "_"

def derive_upos_feats(rec: dict) -> tuple[str, str]:
    primary = rec.get("primary") or {}
    upos = safe(primary.get("upos"))
    feats = safe(primary.get("feats"))

    # Back-compat: sometimes tag contains UPOS|FEATS
    tag = primary.get("tag")
    if isinstance(tag, str) and "|" in tag and upos == "_":
        parts = tag.split("|", 1)
        if parts and parts[0]:
            upos = parts[0].strip() or "_"
        if len(parts) == 2 and parts[1]:
            feats = parts[1].strip() or "_"

    upos = safe(upos)
    feats = clean_feats(feats if feats != "_" else "")

    return upos, feats

def main():
    if not TOKENS.exists():
        raise FileNotFoundError(f"No existe: {TOKENS}")

    tokens = []
    with TOKENS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.append(json.loads(line))

    # Build sentences: split on SENT_FINAL tokens that are punctuation
    sents = []
    cur = []
    for rec in tokens:
        cur.append(rec)
        tok = (rec.get("token") or "").strip()
        if tok in SENT_FINAL:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for i, sent in enumerate(sents, start=1):
            # reconstruct text with spaces (simple, but ok for now)
            text = " ".join((r.get("token") or "").strip() for r in sent).strip()
            out.write(f"# sent_id = sent-{i:04d}\n")
            out.write(f"# text = {text}\n")

            j = 0
            for rec in sent:
                form = (rec.get("token") or "").strip()
                if not form:
                    continue
                j += 1

                tok_key = safe(rec.get("token_key"))
                lemma = safe((rec.get("primary") or {}).get("lemma_key"))

                upos, feats = derive_upos_feats(rec)

                # MISC: always include TokKey; use '_' if missing
                misc = f"TokKey={tok_key}"

                # columns: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC

                # --- FEATS_SANITIZER_V2 ---
                # Arregla el bug de pegado: "Case=NoTokKey=συ" / "Case=NoTokKey=εγω"
                # 1) Si TokKey se pegó dentro del valor Case, recorta en TokKey
                feats = re.sub(r"(Case=[A-Za-z]+)TokKey=[^|\s]+", r"\1", feats)

                # 2) Normaliza Case=No* y Case=Nomm -> Case=Nom
                feats = re.sub(r"Case=No[^|\s]*", "Case=Nom", feats)
                feats = re.sub(r"Case=Nom+m", "Case=Nom", feats)

                # 3) Elimina cualquier TokKey colado en feats (por si aparece separado)
                feats = re.sub(r"TokKey=[^|\s]+", "", feats)

                # 4) Limpieza de separadores
                feats = re.sub(r"\|{2,}", "|", feats).strip("|")
                if feats == "":
                    feats = "_"
                # --- /FEATS_SANITIZER_V2 ---

                out.write(f"{j}\t{form}\t{lemma}\t{upos}\t_\t{feats}\t_\t_\t_\t{misc}\n")
            out.write("\n")

    print(f"[OK] wrote: {OUT} sents={len(sents)} tokens={len(tokens)}")

if __name__ == "__main__":
    main()
