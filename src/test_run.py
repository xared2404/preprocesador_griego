# src/test_run.py
from src.preprocesador.lexicon import attic_tragedy_frame
from src.preprocesador.pipeline import CognitivePreprocessor

text = """
Antígona desafía la ley del tirano por una justicia superior.
νόμος δίκη πόλις
"""

frame = attic_tragedy_frame()
p = CognitivePreprocessor(frame)
res = p.run(text)

print("FRAME:", res.frame_name)
print("SCORES:", res.scores)
print("DOMINANT:", res.dominant_dimension)
print("EXPLANATION:", res.explanation["note"])
