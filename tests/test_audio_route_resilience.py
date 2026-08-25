"""Deployment-resilience regression tests for the audio endpoints.

Context: the native-audio endpoints are wired into ``app.main`` via a top-level
import of the optional audio stack (openSMILE + audio I/O, see
``requirements-audio.txt``). If that import were allowed to abort module load,
a deployment whose audio extras failed to install would fail to start
``uvicorn app.main:app`` at all — its health check would never pass and the
platform would keep serving the previous (audio-less) build, so the audio
routes would appear to "vanish" (HTTP 404) even though they are committed.

These tests pin the resilient contract instead:

* ``app.main`` imports even when the audio extras are unavailable;
* the audio routes are *always* registered in the OpenAPI schema;
* the core tabular service stays healthy (``/api/health`` -> 200);
* the audio endpoints degrade to a clean ``503`` (never 404, never a crash).

The unavailable-stack case is simulated in a subprocess (blocking the audio
modules via ``sys.modules[name] = None``) so it never pollutes the module cache
of the rest of the suite. The available-stack behaviour (200 responses) is
covered by ``tests/test_audio_api.py``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_and_registers_audio_routes_without_audio_stack() -> None:
    """App must boot, expose the audio routes, and 503 them when extras are absent."""
    code = textwrap.dedent(
        """
        import sys

        # Simulate a deployment where the optional audio extras never installed:
        # force `import <extra>` to raise ImportError. This is exactly what would
        # happen on a host that ran requirements.txt but not requirements-audio.txt.
        for _name in ("audiofile", "opensmile", "soundfile", "audinterface"):
            sys.modules[_name] = None

        import app.main as main

        # 1) The module imported despite the missing extras.
        assert main._AUDIO_AVAILABLE is False, "expected audio stack to be flagged unavailable"

        # 2) Both audio routes are still registered in the schema (not dropped).
        paths = set(main.app.openapi()["paths"])
        assert "/api/audio/info" in paths, sorted(paths)
        assert "/api/audio/predict" in paths, sorted(paths)

        from fastapi.testclient import TestClient

        client = TestClient(main.app)

        # 3) Core tabular service is unaffected and healthy.
        health = client.get("/api/health")
        assert health.status_code == 200, health.text
        assert health.json()["n_features"] == 753

        # 4) Audio endpoints degrade cleanly to 503 (not 404, not a 500 crash),
        #    and leak no internal path in the message.
        info = client.get("/api/audio/info")
        assert info.status_code == 503, (info.status_code, info.text)
        detail = info.json().get("detail", "")
        assert "C:" not in detail and "/c/" not in detail and "Traceback" not in detail

        print("RESILIENCE_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "RESILIENCE_OK" in result.stdout, result.stdout
