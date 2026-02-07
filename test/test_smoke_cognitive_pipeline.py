from preprocesador.lexicon import attic_tragedy_frame
from preprocesador.pipeline import CognitivePreprocessor


def test_cognitive_preprocessor_smoke():
    pre = CognitivePreprocessor(frame=attic_tragedy_frame)
    res = pre.run("ὁ ἄνθρωπος καὶ ἡ γυνή .")

    # Frame id should be stable
    assert res.frame == "attic_tragedy_v0"

    # Scores should exist and be a dict with >0 dimensions
    assert isinstance(res.scores, dict) and len(res.scores) > 0

    # Dominant dimension is optional: valid outcomes are
    # - None if no evidence / all zeros
    # - one of the score keys otherwise
    if res.dominant_dimension is not None:
        assert res.dominant_dimension in res.scores
    else:
        # if None, we expect "no evidence" behavior
        assert all(v == 0.0 for v in res.scores.values())

    # Normalization + token/lemma shape invariants
    assert isinstance(res.normalized_text, str) and len(res.normalized_text) > 0
    assert isinstance(res.tokens, list) and len(res.tokens) > 0
    assert isinstance(res.lemmas, list) and len(res.lemmas) == len(res.tokens)
