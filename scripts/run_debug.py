from preprocesador.lexicon import attic_tragedy_frame
from preprocesador.pipeline import CognitivePreprocessor

def main():
    pre = CognitivePreprocessor(frame=attic_tragedy_frame)
    res = pre.run("ὁ ἄνθρωπος καὶ ἡ γυνή .")

    print("FRAME:", res.frame)
    print("NOTE:", res.note)
    print("NORMALIZED:", res.normalized_text)
    print("TOKENS:", res.tokens)
    print("LEMMAS:", res.lemmas)

    top = sorted(res.scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    print("TOP_SCORES:", top)
    print("DOMINANT:", res.dominant_dimension)

if __name__ == "__main__":
    main()
