"""Tests for the production audio API (:mod:`app.audio_inference` + endpoints).

These drive the two new audio endpoints through FastAPI's ``TestClient`` with
tiny synthetic WAVs written on the fly — the ~606 MB MDVR-KCL corpus is never
required. The whole module is skipped if the audio stack is absent so a
Phase-1-only environment stays green.

Guard rails:
* Synthetic audio is asserted only on response *structure*, HTTP status, and
  determinism — never on a predicted class label (a sine wave is out of the
  model's validated domain, and the endpoint is expected to say so).
* Every rejectable input (empty, oversize, wrong extension, non-WAV bytes,
  too short, missing field) is checked for its specific status code.
* The pre-existing 753-feature tabular API is re-checked here to prove the new
  audio surface did not regress it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("opensmile")
pytest.importorskip("audiofile")
sf = pytest.importorskip("soundfile")

from fastapi.testclient import TestClient  # noqa: E402

from app import audio_inference  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------- helpers


def wav_bytes(dur_s=2.0, sr=44100, freq=170.0, subtype="PCM_24", channels=1):
    """A small deterministic voiced-ish tone encoded as WAV bytes in memory."""
    import io

    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    # A couple of harmonics + gentle vibrato so eGeMAPS yields finite F0/jitter
    # features rather than degenerate values, while staying fully deterministic.
    sig = (
        0.30 * np.sin(2 * np.pi * freq * t)
        + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.05 * np.sin(2 * np.pi * 3 * freq * t)
    ).astype(np.float32)
    data = sig if channels == 1 else np.stack([sig] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, subtype=subtype, format="WAV")
    return buf.getvalue()


def post_audio(data, filename="sample.wav", content_type="audio/wav"):
    return client.post("/api/audio/predict", files={"file": (filename, data, content_type)})


# ---------------------------------------------------------------- /api/audio/info


def test_audio_info_endpoint():
    body = client.get("/api/audio/info").json()
    assert body["tag"] == "egemaps-lr@1.0.0"
    assert body["feature_extractor"]["n_features"] == 88
    assert 0.0 < body["operating_threshold"] < 1.0
    assert 0.0 <= body["headline_metric"]["mean"] <= 1.0
    assert "repeated" in body["headline_metric"]["name"].lower()
    assert body["max_upload_mb"] > 0
    assert "not a medical" in body["disclaimer"].lower()


# ---------------------------------------------------------------- valid prediction


def test_audio_predict_valid_wav_structure():
    body = post_audio(wav_bytes(dur_s=2.0)).json()

    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_tag"] == "egemaps-lr@1.0.0"
    assert body["n_features"] == 88
    assert 0.0 < body["threshold"] < 1.0

    assert len(body["explanation"]["top_contributions"]) == 8
    assert set(body["input_domain"]) >= {"in_domain", "max_abs_z", "note"}

    audio = body["audio"]
    assert audio["filename"] == "sample.wav"
    assert audio["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert audio["container_format"].upper() == "WAV"
    assert "not a medical" in body["disclaimer"].lower()


def test_audio_predict_is_deterministic():
    data = wav_bytes(dur_s=2.0)
    p1 = post_audio(data).json()["probability"]
    p2 = post_audio(data).json()["probability"]
    assert p1 == p2


def test_audio_predict_resamples_nonnative_rate():
    body = post_audio(wav_bytes(dur_s=1.5, sr=16000, subtype="PCM_16")).json()
    assert body["audio"]["original_sample_rate"] == 16000
    assert body["audio"]["sample_rate"] == 44100
    assert body["audio"]["resampled"] is True


# ---------------------------------------------------------------- rejections


def test_audio_predict_rejects_empty_upload():
    r = post_audio(b"", filename="empty.wav")
    assert r.status_code == 400


def test_audio_predict_rejects_non_wav_extension():
    r = post_audio(wav_bytes(), filename="clip.mp3")
    assert r.status_code == 415


def test_audio_predict_rejects_non_wav_bytes():
    r = post_audio(b"this is definitely not audio" * 10, filename="fake.wav")
    assert r.status_code == 400


def test_audio_predict_rejects_too_short():
    r = post_audio(wav_bytes(dur_s=0.2), filename="short.wav")
    assert r.status_code == 422


def test_audio_predict_requires_file_field():
    assert client.post("/api/audio/predict").status_code == 422


def test_audio_predict_rejects_oversize(monkeypatch):
    monkeypatch.setattr(audio_inference, "MAX_UPLOAD_BYTES", 1024)
    r = post_audio(wav_bytes(dur_s=2.0), filename="big.wav")
    assert r.status_code == 413


def test_error_messages_leak_no_paths():
    """Rejection messages must never expose internal filesystem paths."""
    detail = post_audio(b"nope", filename="fake.wav").json()["detail"]
    assert "\\" not in detail and "/" not in detail


# ---------------------------------------------------------------- no tabular regression


def test_tabular_health_unchanged():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] == 753


def test_tabular_predict_unchanged():
    body = client.post("/api/predict", json={"features": {}, "model": "attention"}).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
