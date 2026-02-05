from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("reports/out.csv")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No existe {CSV_PATH}. Primero corre run_multiframe con --csv reports/out.csv")

    df = pd.read_csv(CSV_PATH)

    # --- 1) matriz Frames × Dimensiones (scores) ---
    score_cols = sorted([c for c in df.columns if c.startswith("score__")])
    if not score_cols:
        raise ValueError("No encontré columnas score__*. Revisa reports/out.csv")

    frames = df["frame"].tolist()
    dims = [c.replace("score__", "") for c in score_cols]

    M = df[score_cols].to_numpy()

    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.xticks(range(len(dims)), dims, rotation=45, ha="right")
    plt.yticks(range(len(frames)), frames)
    plt.colorbar(label="Lexical score")
    plt.title("Frames × Dimensiones (lexical scores)")
    plt.tight_layout()
    out1 = OUT_DIR / "heatmap_frames_dims.png"
    plt.savefig(out1, dpi=200)
    print(f"[OK] {out1}")

    # --- 2) barras: dominante por frame (lexical) ---
    # si hay empate, ya lo resolviste por max en pipeline; aquí lo tomamos del CSV
    plt.figure()
    dom = df["dominant_dimension"].fillna("")
    counts = dom.value_counts()
    counts = counts[counts.index != ""]

    if len(counts) > 0:
        counts.plot(kind="bar")
        plt.title("Dominant dimension count (across frames)")
        plt.xlabel("Dimension")
        plt.ylabel("Count")
        plt.tight_layout()
        out2 = OUT_DIR / "dominant_dimension_counts.png"
        plt.savefig(out2, dpi=200)
        print(f"[OK] {out2}")
    else:
        print("[WARN] dominant_dimension vacío en CSV (¿pipeline no lo está exportando?)")

    # --- 3) matched forms: tabla rápida (para auditoría) ---
    form_cols = sorted([c for c in df.columns if c.startswith("forms__")])
    if form_cols:
        audit = df[["frame", "dominant_dimension"] + form_cols]
        out3 = OUT_DIR / "matched_forms_audit.csv"
        audit.to_csv(out3, index=False, encoding="utf-8")
        print(f"[OK] {out3}")


if __name__ == "__main__":
    main()
