"""Guard: every Python module under app/ and src/ must parse.

app.main imports the optional audio stack inside a broad try/except so the
core tabular service survives environments without the audio extras. The
downside is that a SyntaxError in an audio module silently degrades the
service (audio endpoints 503, nothing in the logs, CI green). This test
parses every module with ast so no syntax error can ever ship unnoticed
again, regardless of which optional dependencies are installed.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _modules() -> list[Path]:
    return sorted(p for d in ("app", "src") for p in (ROOT / d).rglob("*.py"))


def test_modules_exist() -> None:
    assert _modules(), "expected Python modules under app/ and src/"


def test_all_modules_parse() -> None:
    bad = []
    for path in _modules():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            bad.append(f"{path.relative_to(ROOT)}: line {exc.lineno}: {exc.msg}")
    assert not bad, "syntax errors found:\n" + "\n".join(bad)
