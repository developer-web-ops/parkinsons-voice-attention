"""Configuration for the Phase 2D compact-CNN comparison (documented, not implicit).

Every choice that affects the CNN's inputs, architecture or training lives here as
an explicit, serialisable value so a result can be traced to the exact configuration
that produced it (mirroring :mod:`src.audio.config` for the eGeMAPS baseline). None
of these values are tuned on held-out test data.

Design notes (documented before implementation, Phase 2D spec):

* ``sample_rate = 16000`` — a compact log-Mel CNN does not benefit from the native
  44.1 kHz the way eGeMAPS pitch/formant descriptors do; 16 kHz covers the speech
  band (0-8 kHz) and keeps spectrograms ~3x smaller. This is a *deliberate* departure
  from the eGeMAPS default (44.1 kHz), recorded here rather than assumed. Resampling
  reuses the deterministic polyphase path in :mod:`src.audio.preprocess`.
* STFT ``n_fft=400`` (25 ms), ``hop_length=160`` (10 ms), ``win_length=400``, Hann
  window — the canonical 16 kHz speech-frame settings.
* ``n_mels=64``, ``fmin=20``, ``fmax=8000`` — a compact Mel resolution. The Mel scale
  is HTK (``2595*log10(1+f/700)``); the triangular filterbank is built in
  :mod:`src.audio.spectrogram` (no torchaudio / librosa dependency is added).
* ``spectrogram_norm`` — per-Mel-bin z-score whose mean/std are fit on the
  *inner-training* windows of each fold only, then applied to val/test. This is the
  spectrogram analogue of the baseline's fold-local ``StandardScaler`` and is the
  mechanism that keeps normalisation leakage-safe (Phase 2D requirement 4).
* Windowing — recordings here are long (min 73 s, mean 141 s), so each is segmented
  into fixed ``window_seconds=4.0`` s windows; ``max_windows_per_recording=8`` evenly
  spaced non-overlapping windows are taken (every recording has >=18 possible, so all
  contribute exactly 8, keeping recordings balanced and compute bounded). A recording
  shorter than one window is zero-padded to one window (none in this corpus).
* Architecture / training — a deliberately tiny CNN (~6k params) for 37 subjects, with
  class-weighted BCE, early stopping on inner-val loss, Platt calibration and a max-F1
  threshold chosen on inner-val (identical philosophy to the frozen baseline).

Choosing these describes *what the CNN sees and how it is trained*; it makes no
clinical claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CnnConfig:
    """Immutable description of the raw-WAV -> log-Mel -> compact-CNN pipeline."""

    # --- audio front end -----------------------------------------------------
    sample_rate: int = 16000
    mono: bool = True
    # STFT / Mel
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    window: str = "hann"
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8000.0
    log_offset: float = 1e-6
    mel_scale: str = "htk"  # 2595*log10(1+f/700)
    spectrogram_norm: str = "per_bin_train_zscore"  # fit on inner-train windows only

    # --- windowing / cropping ------------------------------------------------
    window_seconds: float = 4.0
    max_windows_per_recording: int = 8
    pad_mode: str = "zero"  # applied only if a recording is shorter than one window
    min_duration_s: float = 0.5
    max_duration_s: float = 600.0

    # --- model ---------------------------------------------------------------
    conv_channels: tuple[int, int, int] = (8, 16, 32)
    kernel_size: int = 3
    dropout: float = 0.5

    # --- optimisation --------------------------------------------------------
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 40
    early_stopping_patience: int = 8
    class_weighted_loss: bool = True

    # --- calibration / threshold (mirrors the frozen baseline) ---------------
    calibration: str = "platt_on_inner_val_recording"
    threshold_selection: str = "max_f1_on_inner_val_recording"

    # --- evaluation ----------------------------------------------------------
    n_splits: int = 5
    n_repeats: int = 5  # compute-bounded; the LR reference uses 50 (fast)
    base_seed: int = 42

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.fmax > self.sample_rate / 2:
            raise ValueError("fmax must not exceed the Nyquist frequency")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.max_windows_per_recording < 1:
            raise ValueError("max_windows_per_recording must be >= 1")
        if self.mel_scale != "htk":
            raise ValueError(f"unsupported mel_scale: {self.mel_scale!r}")

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.sample_rate)

    @property
    def n_frames(self) -> int:
        """STFT frames for one window (center=True adds one)."""
        return self.window_samples // self.hop_length + 1

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CNN_CONFIG = CnnConfig()
