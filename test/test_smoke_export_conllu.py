import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / ".venv" / "bin" / "python"
SCRIPT = REPO_ROOT / "scripts" / "export_conllu.py"
OUT = REPO_ROOT / "outputs" / "parsed.conllu"

def run_export() -> str:
    """
    Runs your end-to-end command:
      ./.venv/bin/python scripts/export_conllu.py
    and returns stdout+stderr for assertions/debugging.
    """
    proc = subprocess.run(
        [str(PY), str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"export_conllu.py failed.\n{out}"
    return out

def read_conllu(path: Path) -> str:
    assert path.exists(), f"Expected output missing: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), "Output file is empty"
    return text

def test_end_to_end_produces_conllu_shape():
    run_export()
    conllu = read_conllu(OUT)

    # Basic CoNLL-U shape checks
    assert "# sent_id =" in conllu, "Missing sent_id headers"
    assert "# text =" in conllu, "Missing text headers"

    # Token lines: should have 10 tab-separated columns
    token_lines = [
        line for line in conllu.splitlines()
        if line and not line.startswith("#")
    ]
    assert token_lines, "No token lines found"

    for line in token_lines[:50]:  # cheap smoke bound
        cols = line.split("\t")
        assert len(cols) == 10, f"Not 10 columns: {line}"
        assert cols[0].isdigit(), f"ID not numeric: {line}"
        assert cols[1] != "", f"FORM empty: {line}"
        assert "TokKey=" in cols[9], f"TokKey missing in MISC: {line}"

def test_export_is_deterministic_for_same_input():
    # Run twice and compare full outputs byte-for-byte.
    run_export()
    first = read_conllu(OUT)

    run_export()
    second = read_conllu(OUT)

    assert first == second, "Output is not deterministic across runs"
