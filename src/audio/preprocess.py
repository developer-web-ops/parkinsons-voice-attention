"""Deterministic audio preprocessing: load -> mono -> resample -> normalize -> silence.

Every transform is a pure function of the input signal and the :class:`AudioConfig`,
so the same WAV always yields the same processed signal (a prerequisite for the
"feature extraction may be done offline" allowance in the Phase 2B spec: the step
is deterministic and unsupervised, so it cannot leak label information).

Nothing here decides train/test membership or touches labels. Duration gating
returns an explicit reason string instead of raising, so callers can *report*
every excluded recording rather than silently dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass

import audiofile
import numpy as np
from scipy.signal import resample_poly

from src.audio.config import DEFAULT_CONFIG, AudioConfig


@dataclass(frozen=True)
class ProcessedAudio:
    """A preprocessed signal plus the provenance needed to explain it."""

    signal: np.ndarray  # float32, 1-D, at ``sample_rate``
    sample_rate: int
    duration_s: float
    orig_sample_rate: int
    orig_channels: int
    resampled: bool
    normalized: bool


def load_wav(path) -> tuple[np.ndarray, int]:
    """Read a WAV as float32. Returns ``(signal, sample_rate)``.

    ``audiofile.read`` returns a 1-D array for mono and ``(channels, samples)``
    for multi-channel; both are handled by :func:`to_mono`. 24-bit PCM (the
    MDVR-KCL format) is decoded to float32 in [-1, 1].
    """
    signal, sr = audiofile.read(path, always_2d=False)
    return np.asarray(signal, dtype=np.float32), int(sr)


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Collapse to a single channel by averaging (no-op if already mono)."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        return signal
    # audiofile lays multi-channel out as (channels, samples).
    return signal.mean(axis=0, dtype=np.float32)


def channel_count(signal: np.ndarray) -> int:
    return 1 if signal.ndim == 1 else int(signal.shape[0])


def resample_to(signal: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample to ``target_sr``. Exact no-op when ``sr == target_sr``."""
    if sr == target_sr:
        return np.asarray(signal, dtype=np.float32)
    from math import gcd

    g = gcd(int(sr), int(target_sr))
    up, down = target_sr // g, sr // g
    out = resample_poly(signal.astype(np.float64), up, down)
    return out.astype(np.float32)


def peak_normalize(signal: np.ndarray, peak_dbfs: float) -> np.ndarray:
    """Scale so the largest absolute sample sits at ``peak_dbfs`` dBFS.

    Silent signals are returned unchanged (no division by zero).
    """
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak == 0.0:
        return np.asarray(signal, dtype=np.float32)
    target = 10.0 ** (peak_dbfs / 20.0)
    return (signal * (target / peak)).astype(np.float32)


def trim_edges(signal: np.ndarray, threshold_dbfs: float) -> np.ndarray:
    """Trim leading/trailing samples below ``threshold_dbfs`` relative to peak.

    Only used when ``config.silence == "trim_edges"``. Interior pauses are kept.
    """
    if signal.size == 0:
        return signal
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        return signal
    thresh = peak * (10.0 ** (threshold_dbfs / 20.0))
    loud = np.flatnonzero(np.abs(signal) >= thresh)
    if loud.size == 0:
        return signal
    return signal[loud[0] : loud[-1] + 1]


def normalize(signal: np.ndarray, config: AudioConfig) -> tuple[np.ndarray, bool]:
    if config.normalization == "peak":
        return peak_normalize(signal, config.peak_dbfs), True
    return np.asarray(signal, dtype=np.float32), False  # "none"


def apply_silence(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    if config.silence == "trim_edges":
        return trim_edges(signal, config.silence_threshold_dbfs)
    return signal  # "none"


def preprocess_signal(
    signal: np.ndarray, sr: int, config: AudioConfig = DEFAULT_CONFIG
) -> ProcessedAudio:
    """Apply the full deterministic chain to an in-memory signal."""
    orig_channels = channel_count(signal)
    orig_sr = int(sr)

    mono = to_mono(signal) if config.mono else np.asarray(signal, dtype=np.float32)
    resampled_sig = resample_to(mono, orig_sr, config.target_sample_rate)
    resampled = orig_sr != config.target_sample_rate
    silenced = apply_silence(resampled_sig, config)
    normalized_sig, did_norm = normalize(silenced, config)

    duration = float(len(normalized_sig) / config.target_sample_rate)
    return ProcessedAudio(
        signal=normalized_sig,
        sample_rate=config.target_sample_rate,
        duration_s=duration,
        orig_sample_rate=orig_sr,
        orig_channels=orig_channels,
        resampled=resampled,
        normalized=did_norm,
    )


def preprocess_file(path, config: AudioConfig = DEFAULT_CONFIG) -> ProcessedAudio:
    signal, sr = load_wav(path)
    return preprocess_signal(signal, sr, config)


def duration_exclusion_reason(processed: ProcessedAudio, config: AudioConfig) -> str | None:
    """Return an exclusion reason if the duration is out of band, else ``None``."""
    if processed.duration_s < config.min_duration_s:
        return f"too_short ({processed.duration_s:.3f}s < {config.min_duration_s}s)"
    if processed.duration_s > config.max_duration_s:
        return f"too_long ({processed.duration_s:.3f}s > {config.max_duration_s}s)"
    return None
