# Preprocesador Cognitivo para Texto Griego

## Quickstart

## 1) Create environment

'''python -m venv .venv source .venv/bin/activate pip install -r requirements.txt

### 2) Run the pipeline on a minimal example

# Example (adjust to your script names)

./.venv/bin/python scripts/export_conllu.py
'''

### 3) inspect output
```sed -n '1,120p' outputs/parsed.conllu

### Expected shape:
	•	# sent_id = ... ensure stable sentence ids
	•	# text = ... original sentence
	•	10-column CoNLL-U rows
	•	MISC includes TokKey=...

### Project boundary (important)

This repo implements the pre-NLP cognitive preprocessing layer:
	•	deterministic normalization
	•	deterministic sentence segmentation
	•	deterministic tokenization
	•	CoNLL-U shaped output (parsing fields may be _)

Parsing (Stanza/UDPipe) is a downstream module and must not redefine token boundaries.
See: docs/preprocessor_contract.md.



Este proyecto explora el diseño de un **preprocesador cognitivo-hermenéutico**
para textos griegos antiguos (especialmente tragedia ática).

El objetivo no es limpiar texto, sino **configurar el espacio de sentido**
previo a la modelización NLP.

## Estructura

- `src/preprocesador/`
  - `frame.py`: marcos interpretativos
  - `lexicon.py`: léxicos ponderados
  - `pipeline.py`: integración cognitiva
- `data/`: textos fuente
- `notebooks/`: exploración y pruebas
- `docs/`: notas teóricas

## Estado
Proyecto en fase fundacional.


## Quickstart (Mac / zsh)

```bash
python -m venv .venv
source .venv/bin/activate

# IMPORTANT: pip<24.1 needed to install some HF/spaCy wheels with nonstandard filenames
python -m pip install "pip<24.1"

python -m pip install -r requirements.txt

# HuggingFace token (recommended)
export HF_TOKEN="hf_...."

# Run final ensemble
python -m src.run_pipeline_final \
  --in data/corpus/antigone_grc_excerpt.txt \
  --frame frames/attic_tragedy.yaml \
  --rules rules/rules.yaml \
  --json reports/final_antigone.json \
  --csv reports/final_antigone.csv
