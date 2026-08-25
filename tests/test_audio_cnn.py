"""Phase 2D tests: log-Mel front end, compact CNN, subject-grouped CNN CV, orchestrator.

Everything runs on tiny synthetic signals and feature frames — never the ~606 MB
MDVR-KCL corpus and never a WAV on disk (a fake in-memory loader supplies signals).
The module skips cleanly when the audio/torch stack is absent (``src.audio.cnn_cv``
imports ``src.audio.baseline_cv`` -> openSMILE).

Coverage: HTK Mel scale round-trip and filterbank; windowing branches + zero-pad +
determinism; log-Mel shape/determinism; CNN parameter count (small) and forward
shape; train-only per-bin standardisation; single-fold fit with no leakage; pooled
OOF coverage + determinism; repeated CV structure/determinism; identical-fold paired
CNN-vs-LR comparison; canonical bootstrap CIs; and the orchestrator writing only
under its out_dir with an honest verdict when the frozen reference is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# cnn_cv imports baseline_cv -> opensmile; the model/spectrogram need torch.
pytest.importorskip("torch")
pytest.importorskip("opensmile")

import torch

from src.audio.cnn_config import CnnConfig
from src.audio.cnn_cv import (
    PAIRED_METRICS,
    RawSignalLoader,
    _apply_standardizer,
    _fit_standardizer,
    build_window_cache,
    canonical_bootstrap_cnn,
    fit_cnn_fold,
    paired_cnn_vs_lr,
    reference_oof_cnn,
    repeated_cnn_cv,
)
from src.audio.cnn_model import SmallAudioCNN, count_parameters
from src.audio.cnn_phase2d import parameter_report, run
from src.audio.dataset import build_dataset
from src.audio.spectrogram import (
    MelSpectrogram,
    extract_windows,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    window_starts,
)
from src.data import assert_no_group_leakage, cv_splits, inner_subject_split

# Small, fast config for the whole module (real defaults are validated separately).
TEST_CONFIG = CnnConfig(
    sample_rate=4000,
    n_fft=128,
    hop_length=64,
    win_length=128,
    n_mels=16,
    fmin=20.0,
    fmax=2000.0,
    window_seconds=0.5,
    max_windows_per_recording=4,
    max_epochs=12,
    early_stopping_patience=4,
    batch_size=16,
    n_splits=3,
    n_repeats=2,
)


# ----------------------------------------------------------------------------- helpers


def make_frame_and_loader(rng, *, n_per_class=6, recs=2, sr=4000, dur_s=1.0, n_feat=8,
                          separable=True):
    """Synthetic recordings separable by label in BOTH audio and eGeMAPS-like features.

    Audio: PD recordings carry a high tone, HC a low tone, so their log-Mel spectra
    differ (learnable by the CNN). Features: ``feat_i`` shifts with label (learnable
    by the LR baseline). The loader returns the in-memory signal for a ``relpath``.
    """
    rows, signals = [], {}
    t = np.arange(int(sr * dur_s)) / sr
    sid = 1
    for label in (0, 1):
        for _ in range(n_per_class):
            gk = f"S{sid:02d}"
            for r in range(recs):
                relpath = f"{gk}_{r}.wav"
                freq = (1200.0 if label else 250.0) if separable else 500.0
                sig = 0.3 * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.005, t.shape)
                signals[relpath] = sig.astype(np.float32)
                row = {
                    "subject_id": sid,
                    "group_key": gk,
                    "label": label,
                    "cohort_folder": "PD" if label else "HC",
                    "cohort_file": "pd" if label else "hc",
                    "task": "ReadText" if r == 0 else "SpontaneousDialogue",
                    "filename": relpath,
                    "relpath": relpath,
                    "duration_s": dur_s,
                    "orig_sample_rate": sr,
                    "orig_channels": 1,
                }
                feats = rng.normal(label * (2.0 if separable else 0.0), 1.0, size=n_feat)
                row.update({f"feat_{i:02d}": float(v) for i, v in enumerate(feats)})
                rows.append(row)
            sid += 1
    frame = pd.DataFrame(rows)

    def loader(relpath):
        return signals[relpath]

    return frame, loader


# ----------------------------------------------------------------------------- spectrogram


def test_mel_scale_round_trip():
    hz = np.array([0.0, 100.0, 440.0, 1000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz)), hz, rtol=1e-6, atol=1e-6)


def test_mel_filterbank_shape_and_bounds():
    fb = mel_filterbank(sample_rate=16000, n_fft=400, n_mels=64, fmin=20.0, fmax=8000.0)
    assert fb.shape == (64, 201)  # n_fft//2 + 1
    assert np.all(fb >= 0.0)
    assert fb.max() <= 1.0 + 1e-6  # HTK triangles peak at 1.0
    assert np.all(fb.sum(axis=1) > 0.0)  # no empty filter


def test_mel_filterbank_rejects_supra_nyquist_fmax():
    with pytest.raises(ValueError):
        mel_filterbank(sample_rate=16000, n_fft=400, n_mels=64, fmin=20.0, fmax=9000.0)


def test_window_starts_branches_are_non_overlapping():
    # short -> single window at 0
    assert window_starts(50, 100, 4) == [0]
    # few possible -> consecutive, spaced by exactly one window
    assert window_starts(300, 100, 4) == [0, 100, 200]
    # many possible -> evenly spread, spacing >= one window (non-overlap)
    starts = window_starts(1000, 100, 4)
    assert len(starts) == 4 and starts[0] == 0 and starts[-1] == 900
    assert all(starts[i + 1] - starts[i] >= 100 for i in range(len(starts) - 1))


def test_extract_windows_shape_padding_and_determinism():
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 1, size=1000).astype(np.float32)
    w = extract_windows(sig, win=100, max_windows=4)
    assert w.shape == (4, 100)
    np.testing.assert_array_equal(w, extract_windows(sig, 100, 4))
    # signal shorter than a window is zero-padded to one window
    short = extract_windows(sig[:30], win=100, max_windows=4)
    assert short.shape == (1, 100)
    np.testing.assert_array_equal(short[0, :30], sig[:30])
    assert np.all(short[0, 30:] == 0.0)


def test_log_mel_shape_and_determinism():
    mel = MelSpectrogram(TEST_CONFIG)
    sig = 0.2 * np.sin(2 * np.pi * 300 * np.arange(TEST_CONFIG.window_samples) / 4000)
    a = mel(sig.astype(np.float32))
    b = mel(sig.astype(np.float32))
    assert a.shape == (TEST_CONFIG.n_mels, TEST_CONFIG.n_frames)
    assert torch.allclose(a, b)
    assert torch.isfinite(a).all()


# ----------------------------------------------------------------------------- model


def test_cnn_parameter_count_is_small():
    model = SmallAudioCNN(TEST_CONFIG)
    n = count_parameters(model)
    assert 0 < n < 10_000  # deliberately tiny for 37 subjects


def test_default_cnn_param_count_documented_range():
    # The reported "~6k parameters" claim for the real config.
    model = SmallAudioCNN(CnnConfig())
    assert 5_000 <= count_parameters(model) <= 7_000


def test_cnn_forward_shape():
    model = SmallAudioCNN(TEST_CONFIG)
    x = torch.zeros(5, 1, TEST_CONFIG.n_mels, TEST_CONFIG.n_frames)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (5, 1)


def test_parameter_report_structure():
    rep = parameter_report(TEST_CONFIG)
    assert rep["n_parameters"] == sum(rep["per_parameter_tensor"].values())
    assert rep["output_shape"] == [1, 1]
    assert rep["float32_size_bytes"] == rep["n_parameters"] * 4


# ----------------------------------------------------------------------------- cache / norm


def test_build_window_cache_shapes_and_mapping():
    rng = np.random.default_rng(1)
    frame, loader = make_frame_and_loader(rng, n_per_class=3, recs=2)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    n_rec = len(frame)
    assert cache.n_recordings == n_rec
    assert cache.logmel.shape[1:] == (1, TEST_CONFIG.n_mels, TEST_CONFIG.n_frames)
    # every window maps to a valid recording; counts per recording are positive
    assert set(np.unique(cache.rec_of_win)) == set(range(n_rec))
    assert cache.y_win.shape[0] == cache.logmel.shape[0]
    # window label matches its recording's label
    for i in range(n_rec):
        assert set(cache.y_win[cache.rec_of_win == i]) == {int(frame.iloc[i]["label"])}


def test_standardizer_is_fit_on_training_windows_only():
    rng = np.random.default_rng(2)
    frame, loader = make_frame_and_loader(rng, n_per_class=3, recs=2)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    mean, std = _fit_standardizer(cache.logmel)
    z = _apply_standardizer(cache.logmel, mean, std)
    # standardised training data has ~zero mean and ~unit std per Mel bin
    per_bin_mean = z.mean(dim=(0, 1, 3))
    per_bin_std = z.std(dim=(0, 1, 3))
    assert torch.allclose(per_bin_mean, torch.zeros_like(per_bin_mean), atol=1e-4)
    assert torch.allclose(per_bin_std, torch.ones_like(per_bin_std), atol=1e-2)


# ----------------------------------------------------------------------------- fold / CV


def test_fit_cnn_fold_no_leakage_and_shapes():
    rng = np.random.default_rng(3)
    frame, loader = make_frame_and_loader(rng, n_per_class=6, recs=2)
    data = build_dataset(frame)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    train_all, test = cv_splits(data, n_splits=3, seed=42)[0]
    inner_train, val = inner_subject_split(data, train_all, seed=42)
    assert_no_group_leakage(inner_train, test, data.groups)
    assert_no_group_leakage(val, test, data.groups)

    fold = fit_cnn_fold(cache, data, inner_train, val, test, TEST_CONFIG, seed=42)
    assert set(fold) >= {"test_prob", "threshold", "calibrated", "epochs", "model"}
    assert fold["test_prob"].shape == (len(test),)
    assert 0.05 <= fold["threshold"] <= 0.95
    assert np.all((fold["test_prob"] >= 0) & (fold["test_prob"] <= 1))


def test_reference_oof_cnn_covers_all_and_is_deterministic():
    rng = np.random.default_rng(4)
    frame, loader = make_frame_and_loader(rng, n_per_class=6, recs=2)
    data = build_dataset(frame)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    prob, pred, thr, folds, _cal, epochs = reference_oof_cnn(
        cache, data, n_splits=3, seed=42, config=TEST_CONFIG
    )
    assert prob.shape == (len(data.y),)
    assert not np.isnan(prob).any() and not np.isnan(thr).any()
    assert set(np.unique(pred)) <= {0, 1}
    assert len(folds) == 3 and len(epochs) == 3
    prob2, pred2, *_ = reference_oof_cnn(cache, data, n_splits=3, seed=42, config=TEST_CONFIG)
    np.testing.assert_array_equal(prob, prob2)
    np.testing.assert_array_equal(pred, pred2)


def test_repeated_cnn_cv_structure_and_determinism():
    from src.audio.robustness import REPEAT_METRIC_NAMES

    rng = np.random.default_rng(5)
    frame, loader = make_frame_and_loader(rng, n_per_class=6, recs=2)
    data = build_dataset(frame)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    a = repeated_cnn_cv(cache, data, n_splits=3, n_repeats=2, base_seed=42, config=TEST_CONFIG)
    assert a["seeds"] == [42, 43]
    assert set(a["metrics"]) == set(REPEAT_METRIC_NAMES)
    for stats in a["metrics"].values():
        assert {"n", "mean", "std", "p2_5", "p97_5"} <= set(stats)
        assert stats["n"] == 2
    assert len(a["fold_thresholds"]) == 2 * 3
    b = repeated_cnn_cv(cache, data, n_splits=3, n_repeats=2, base_seed=42, config=TEST_CONFIG)
    assert a["per_repeat"] == b["per_repeat"]


def test_canonical_bootstrap_cnn_has_both_levels_with_cis():
    rng = np.random.default_rng(6)
    frame, loader = make_frame_and_loader(rng, n_per_class=6, recs=2)
    data = build_dataset(frame)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    out = canonical_bootstrap_cnn(cache, data, n_splits=3, seed=42, n_boot=40, config=TEST_CONFIG)
    for level in ("recording_level", "subject_level"):
        assert "roc_auc" in out[level]
        assert {"point", "ci_low", "ci_high"} <= set(out[level]["roc_auc"])
    assert len(out["epochs_per_fold"]) == 3


def test_cnn_learns_on_separable_audio():
    rng = np.random.default_rng(7)
    frame, loader = make_frame_and_loader(rng, n_per_class=8, recs=2, separable=True)
    data = build_dataset(frame)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    out = canonical_bootstrap_cnn(cache, data, n_splits=3, seed=42, n_boot=40, config=TEST_CONFIG)
    # Distinct tones per class -> the CNN should do better than chance.
    assert out["subject_level"]["roc_auc"]["point"] > 0.6


# ----------------------------------------------------------------------------- paired vs LR


def test_paired_cnn_vs_lr_identical_folds_and_structure():
    rng = np.random.default_rng(8)
    frame, loader = make_frame_and_loader(rng, n_per_class=6, recs=2)
    data = build_dataset(frame)
    cache = build_window_cache(frame, TEST_CONFIG, loader)
    paired = paired_cnn_vs_lr(cache, data, n_splits=3, seeds=[42, 43], config=TEST_CONFIG)
    assert paired["seeds"] == [42, 43]
    assert len(paired["per_seed"]) == 2
    for row in paired["per_seed"]:
        assert set(row["cnn"]) == set(PAIRED_METRICS)
        assert set(row["lr"]) == set(PAIRED_METRICS)
    for m in PAIRED_METRICS:
        s = paired["summary"][m]
        assert s["n_seeds"] == 2
        assert 0 <= s["cnn_wins"] <= 2
        assert "higher_is_better" in s


# ----------------------------------------------------------------------------- loader


def test_raw_signal_loader_uses_cnn_audio_config(tmp_path):
    # Construction only — never reads a WAV, never touches the real corpus.
    loader = RawSignalLoader(TEST_CONFIG, raw_dir=tmp_path)
    assert loader.raw_dir == tmp_path
    assert loader.audio_config.target_sample_rate == TEST_CONFIG.sample_rate
    assert loader.audio_config.normalization == "none"
    assert loader.audio_config.silence == "none"


# ----------------------------------------------------------------------------- orchestrator


def test_run_writes_all_artifacts_under_out_dir(tmp_path):
    rng = np.random.default_rng(9)
    frame, loader = make_frame_and_loader(rng, n_per_class=6, recs=2)
    feature_csv = tmp_path / "feat.csv"
    frame.to_csv(feature_csv, index=False)
    out_dir = tmp_path / "cnn"

    summary = run(
        n_repeats=2,
        n_splits=3,
        base_seed=42,
        seed=42,
        n_boot=30,
        config=TEST_CONFIG,
        feature_csv=feature_csv,
        phase2c_dir=tmp_path / "no_phase2c",  # frozen reference absent
        cv_metrics_path=tmp_path / "no_metrics.json",
        out_dir=out_dir,
        loader=loader,
    )

    for fname in ("cnn_config.json", "cnn_metrics.json", "cnn_comparison.json",
                  "cnn_summary.json", "cnn_example.pt"):
        assert (out_dir / fname).exists()
    # verdict is honest even without the frozen reference
    assert "verdict" in summary and "statement" in summary["verdict"]
    assert summary["parameters"] == count_parameters(SmallAudioCNN(TEST_CONFIG))
    metrics = json.loads((out_dir / "cnn_metrics.json").read_text(encoding="utf-8"))
    assert metrics["phase"] == "2D"
    assert metrics["parameters"]["n_parameters"] == summary["parameters"]
    assert metrics["timing"]["device"] == "cpu"
