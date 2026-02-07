# Preprocessor Contract (Pre-NLP Cognitive Layer)

## Purpose
This repository provides a **pre-NLP preprocessing layer** for Ancient Greek texts.
Its goal is to produce **stable, reproducible** segmented output suitable for downstream NLP tools
(e.g., Stanza / UDPipe) **without** committing to a specific syntactic interpretation.

This layer is intentionally conservative: it standardizes what must be standardized and preserves what must be preserved.

---

## Scope

### What this layer DOES
- Loads raw Greek text (Unicode) and/or text extracted from structured sources.
- Performs **deterministic normalization** (explicitly defined below).
- Performs **sentence segmentation** (deterministic rules).
- Performs **tokenization** (deterministic rules).
- Emits **CoNLL-U-compatible output** (may be partial; parsing fields can be blank).
- Produces stable keys (e.g., `TokKey`) for cross-step alignment.

### What this layer DOES NOT do (by design)
- No dependency parsing (HEAD/DEPREL not guaranteed).
- No semantic disambiguation / WSD.
- No editorial decisions about variants or emendations.
- No “smart” post-corrections based on lexica (unless explicitly added as a separate module).

---

## Inputs

### Accepted input types
- Plain text files (`.txt`) containing Greek Unicode.
- Programmatic strings (CLI or library usage).
- Text extracted from XML/TEI (via adapters), as long as it becomes raw Unicode text before entering the core pipeline.

### Encoding assumptions
- UTF-8.
- Input should be valid Unicode; invalid byte sequences are rejected.

### Recommended Unicode form
- The pipeline SHOULD enforce (and/or document) a chosen Unicode normalization form (e.g., NFC).
- The chosen form must be stable and consistent across runs.

---

## Normalization Rules (Deterministic)

This section MUST reflect the current implementation. Update it only when the implementation changes.

### Required invariants
- The pipeline produces the same output given the same input (deterministic).
- Token boundaries do not depend on randomness or model inference.

### Suggested (typical) normalizations (examples)
- Unicode normalization to NFC (or NFKC) — choose one and freeze it.
- Whitespace normalization (collapse multiple spaces, normalize newlines).
- Punctuation normalization (only if explicitly defined and stable).
- Lowercasing (only if explicitly defined; preserve original token form separately if needed).

> NOTE: If you preserve original forms, document where they live (e.g., FORM column vs. TokKey).

---

## Tokenization Contract

### Token definition
A token is a unit emitted as one CoNLL-U row, aligned to the surface text.

### Stable key: `TokKey`
`TokKey` is a deterministic, stable key derived from the token surface form (and possibly normalization rules).

**Invariants**
- Same surface token → same TokKey (under the same normalization configuration).
- TokKey is intended for matching across pipeline stages.

**Recommended practice**
- Keep the original surface form in the FORM column.
- Keep TokKey in MISC (e.g., `TokKey=...`) so downstream tools can preserve it.

---

## Sentence Segmentation Contract

### Sentence definition
A sentence is a unit emitted under a CoNLL-U `# sent_id` header and `# text` line.

**Invariants**
- `sent_id` must be stable per run given same input and same segmentation rules.
- `# text` is the surface sentence string corresponding to the token rows.

---

## Output Format

### CoNLL-U compliance
The output is CoNLL-U shaped:
- 10 columns per token line.
- Comment headers per sentence, at minimum:
  - `# sent_id = ...`
  - `# text = ...`

### Fields
- ID: integer 1..N in sentence
- FORM: token surface form (Greek)
- LEMMA / UPOS / XPOS / FEATS / HEAD / DEPREL / DEPS: may be `_` or blank depending on stage
- MISC: includes at least `TokKey=...`

### Guarantee level
- This layer guarantees: **ID, FORM, sentence boundaries, TokKey**
- Anything else is optional until a downstream annotator/parser step fills it.

---

## Downstream Compatibility

### Intended consumers
- Stanza / UDPipe pipelines consuming tokenized sentences.
- Evaluation scripts that align tokens by `sent_id` + ID + TokKey.
- Later “cognitive” modules operating on stable token streams.

### Forbidden behavior for downstream steps
Downstream steps MUST NOT modify:
- token boundaries
- token order
- TokKey semantics

If a downstream step must retokenize, it must do so in a separate forked output.

---

## Versioning Policy

### Interface freezes
- Any change that affects tokenization, segmentation, or TokKey generation is an **interface-breaking change**.
- Interface-breaking changes require:
  - explicit changelog entry
  - version bump
  - updated fixtures/tests

---

## Minimal Example (Expected Shape)

Example sentence:

# sent_id = sent-0001
# text = ὁ ἄνθρωπος καὶ ἡ γυνή .

Token rows (columns may be `_` except ID/FORM/MISC):

1	ὁ	_	DET	_	_	_	_	_	TokKey=ο
2	ἄνθρωπος	_	_	_	_	_	_	_	TokKey=ανθρωποσ
3	καὶ	_	CCONJ	_	_	_	_	_	TokKey=και
...
