"""Deterministic log-Mel spectrogram front end for the Phase 2D compact CNN.

The Phase 2D spec forbids adding a large pretrained model and keeps torchaudio /
librosa out of the dependency set at this stage. This module therefore builds the
log-Mel representation from primitives that are *already* installed:

* the STFT via :func:`torch.stft` (torch is a Phase 1 dependency), and
* a hand-built HTK triangular Mel filterbank (plain NumPy).

The transform is a pure function of the signal and the :class:`CnnConfig`: the same
waveform always produces the same spectrogram, so — like the eGeMAPS extractor — it
is deterministic and unsupervised and cannot leak label information. All label-aware
fitting (per-bin standardisation, calibration, threshold) happens later, inside each
training fold (:mod:`src.audio.cnn_cv`).

Nothing here reads WAVs or touches labels; signal loading lives in
:mod:`src.audio.cnn_cv` so this module stays trivially unit-testable on synthetic
arrays.
"""

from __future__ import annotations

import numpy as np
import torch

from src.audio.cnn_config import DEFAULT_CNN_CONFIG, CnnConfig


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """HTK Mel scale: ``2595 * log10(1 + f/700)``."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Inverse HTK Mel scale."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Triangular HTK Mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Filters are *not* area-normalised (HTK convention), so each triangle peaks at
    1.0. Deterministic and independent of any data.
    """
    if fmax > sample_rate / 2:
        raise ValueError("fmax must not exceed the Nyquist frequency")
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        left = (fft_freqs - f_left) / (f_center - f_left)
        right = (f_right - fft_freqs) / (f_right - f_center)
        fb[m - 1] = np.clip(np.minimum(left, right), 0.0, None)
    return fb.astype(np.float32)


def window_starts(n_samples: int, win: int, max_windows: int) -> list[int]:
    """Start indices for up to ``max_windows`` fixed-length windows.

    * ``n_samples <= win``            -> a single window at 0 (caller zero-pads).
    * ``n_samples // win <= max``     -> consecutive non-overlapping windows.
    * otherwise                       -> ``max_windows`` windows spread evenly across
      the recording (``linspace`` over valid starts). For this corpus every
      recording has >= 18 possible windows, so this last branch always applies and
      each recording contributes exactly ``max_windows`` windows.
    """
    if n_samples <= win:
        return [0]
    n_possible = n_samples // win
    if n_possible <= max_windows:
        return [k * win for k in range(n_possible)]
    # np.float64 starts -> round to nearest sample and return plain python ints
    return np.linspace(0, n_samples - win, max_windows).round().astype(int).tolist()


def extract_windows(signal: np.ndarray, win: int, max_windows: int) -> np.ndarray:
    """Slice a 1-D signal into ``(n_windows, win)`` fixed-length windows.

    A signal shorter than one window is zero-padded up to ``win`` (documented
    ``pad_mode="zero"``); no recording in this corpus triggers that path.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    n = signal.shape[0]
    if n < win:
        padded = np.zeros(win, dtype=np.float32)
        padded[:n] = signal
        return padded[None, :]
    starts = window_starts(n, win, max_windows)
    return np.stack([signal[s : s + win] for s in starts], axis=0).astype(np.float32)


class MelSpectrogram:
    """Callable, config-bound waveform -> log-Mel transform (deterministic).

    Built once and reused across every window/fold: the Hann window and the Mel
    filterbank are constant tensors, so calling it is a pure STFT + matmul + log.
    """

    def __init__(self, config: CnnConfig = DEFAULT_CNN_CONFIG) -> None:
        self.config = config
        self.window = torch.hann_window(config.win_length, periodic=True)
        fb = mel_filterbank(
            config.sample_rate, config.n_fft, config.n_mels, config.fmin, config.fmax
        )
        self.filterbank = torch.from_numpy(fb)  # (n_mels, n_freqs)

    def __call__(self, signal: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return the ``(n_mels, n_frames)`` log-Mel spectrogram of one window."""
        x = torch.as_tensor(np.asarray(signal, dtype=np.float32).reshape(-1))
        spec = torch.stft(
            x,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)  # (n_freqs, n_frames)
        mel = self.filterbank @ power  # (n_mels, n_frames)
        return torch.log(mel + self.config.log_offset)
