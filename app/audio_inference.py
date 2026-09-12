"""Production audio inference: WAV upload -> Parkinson's probability + explanation.

This is the deployable entry point for the native-audio model. It owns the
file-handling and validation concerns that the numeric core (:mod:`src.audio.production`)
deliberately does not:

    raw bytes / path
      -> size guard
      -> WAV format validation (libsndfile)
      -> deterministic preprocessing (src.audio.preprocess, DEFAULT_CONFIG)
      -> duration guard
      -> eGeMAPSv02 extraction (validated feature ordering)
      -> frozen scaler + calibrated LR + operating threshold
      -> prediction + local explanation + reliability report

Design rules honoured here:

* Training and inference preprocessing are the *same* code path (``DEFAULT_CONFIG``
  + :func:`src.audio.preprocess.preprocess_file`), so there is no train/serve skew.
* Feature ordering is validated against the frozen model, never assumed.
* Inputs are never silently accepted: wrong size, non-WAV, unreadable, too short
  or too long each raise a typed error with a safe, user-facing message.
* No internal filesystem paths, stack traces, or dataset internals are ever put
  into an error message or response — temp files are created and removed here and
  their paths never escape.
"""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

from src.audio.config import DEFAULT_CONFIG
from src.audio.preprocess import duration_exclusion_reason, preprocess_file
from src.audio.production import (
    PRODUCTION_MODEL_TAG,
    get_smile,
    load_bundle,
    model_card_summary,
    predict_from_vector,
)

# Upload size cap (resource guard, independent of the scientific duration bound).
# Overridable via AUDIO_MAX_UPLOAD_MB. Default 25 MB ~= 3 min of 44.1 kHz/24-bit mono.
_DEFAULT_MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = int(
    float(os.environ.get("AUDIO_MAX_UPLOAD_MB", _DEFAULT_MAX_UPLOAD_MB)) * 1024 * 1024
)

_DISCLAIMER = (
    "Research/ML prediction only - NOT a medical diagnosis. This tool has not been "
    "evaluated by any regulatory body and must not be used for clinical decisions."
)


# --- Typed errors (status_code consumed by the API layer) --------------------
class AudioInferenceError(Exception):
    """Base for user-facing audio inference failures. Carries an HTTP status."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.public_message = message


class EmptyUploadError(AudioInferenceError):
    status_code = 400


class AudioTooLargeError(AudioInferenceError):
    status_code = 413


class UnsupportedAudioError(AudioInferenceError):
    status_code = 415


class InvalidAudioError(AudioInferenceError):
    status_code = 400


class AudioDurationError(AudioInferenceError):
    status_code = 422


class FeatureExtractionError(AudioInferenceError):
    status_code = 500


def _safe_name(filename: str | None) -> str:
    """Reduce a user-supplied filename to a harmless display basename."""
    if not filename:
        return "upload.wav"
    base = os.path.basename(str(filename).replace("\\", "/"))
    base = base.strip() or "upload.wav"
    return base[:128]



def _validate_wav(path: Path) -> dict[str, Any]:
    """Confirm the bytes are a readable WAV file."""

    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise InvalidAudioError(
            "The uploaded file could not be read as audio. "
            "Please upload a valid WAV file."
        ) from exc

    container_format = (info.format or "").upper()

    # Accept standard WAV and WAVE_FORMAT_EXTENSIBLE.
    if container_format not in {"WAV", "WAVEX"}:
        raise UnsupportedAudioError(
            f"Unsupported audio container '{info.format or 'unknown'}'. "
            "Please upload a WAV file."
        )

    return {
        "container_format": info.format,
        "subtype": info.subtype,
    }


def warm_up() -> None:
    """Eagerly load + validate the model bundle and openSMILE layout (startup)."""
    load_bundle()
    get_smile()


def model_info() -> dict[str, Any]:
    """Model-card summary for the ``/api/audio/info`` endpoint."""
    info = model_card_summary()
    info["max_upload_mb"] = round(MAX_UPLOAD_BYTES / (1024 * 1024), 1)
    info["accepted_format"] = "WAV (any sample rate; resampled to 44.1 kHz mono internally)"
    info["disclaimer"] = _DISCLAIMER
    return info


def predict_wav_bytes(data: bytes, filename: str | None = None) -> dict[str, Any]:
    """Run the full audio pipeline on in-memory WAV bytes and return the payload.

    Raises a subclass of :class:`AudioInferenceError` (each with a ``status_code``)
    for every rejectable input. On success returns a JSON-serialisable dict with
    the prediction, probability, threshold, model identity, per-feature
    explanation, input-domain reliability report, audio metadata, and disclaimer.
    """
    if data is None or len(data) == 0:
        raise EmptyUploadError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AudioTooLargeError(
            f"File is too large ({len(data) / 1e6:.1f} MB). "
            f"The maximum is {MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )

    safe_name = _safe_name(filename)
    if not safe_name.lower().endswith(".wav"):
        raise UnsupportedAudioError(
            "Only WAV audio files are accepted (filename must end in .wav)."
        )

    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)

        container = _validate_wav(tmp_path)

        try:
            processed = preprocess_file(tmp_path, DEFAULT_CONFIG)
        except AudioInferenceError:
            raise
        except Exception as exc:  # unreadable/corrupt audio
            raise InvalidAudioError(
                "The audio could not be decoded. Please upload a valid, uncorrupted WAV file."
            ) from exc

        reason = duration_exclusion_reason(processed, DEFAULT_CONFIG)
        if reason is not None:
            raise AudioDurationError(f"Audio rejected: {reason}.")

        try:
            vector = _extract_vector(processed)
            except Exception as exc:
                logger.exception("Audio feature extraction failed")
                raise FeatureExtractionError(
                    "Acoustic feature extraction failed for this recording."
                    ) from exc

        result = predict_from_vector(vector)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    result["audio"] = {
        "filename": safe_name,
        "duration_s": round(float(processed.duration_s), 3),
        "sample_rate": int(processed.sample_rate),
        "original_sample_rate": int(processed.orig_sample_rate),
        "original_channels": int(processed.orig_channels),
        "resampled": bool(processed.resampled),
        "container_format": container["container_format"],
    }
    result["model_tag"] = PRODUCTION_MODEL_TAG
    result["disclaimer"] = _DISCLAIMER
    return result


def _extract_vector(processed) -> np.ndarray:
    """Extract the 88-dim eGeMAPSv02 vector using the validated cached extractor."""
    from src.audio.features import extract_vector

    smile = get_smile()
    return extract_vector(processed, smile, DEFAULT_CONFIG)
