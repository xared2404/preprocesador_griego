# Preprocesador Cognitivo para Texto Griego

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
