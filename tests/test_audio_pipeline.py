"""Phase 2B audio-pipeline tests.

These use tiny synthetic WAVs and fake directory trees — they never require the
~606 MB MDVR-KCL dataset. The whole module is skipped if the audio stack
(openSMILE / audiofile / soundfile) is not installed, so a Phase-1-only
environment stays green.

Covered: WAV discovery, tolerant subject-id parsing, the ID22 anomaly,
folder/filename label consistency, sample-rate and channel handling,
deterministic preprocessing, the eGeMAPSv02 schema and 88-feature count,
subject-grouped CV with no leakage, train-only scaling, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The audio stack must be present for this module; skip cleanly otherwise.
pytest.importorskip("opensmile")
pytest.importorskip("audiofile")
sf = pytest.importorskip("soundfile")

from src.audio import discovery, preprocess  # noqa: E402
from src.audio.baseline_cv import (  # noqa: E402
    AUDIO_METRIC_NAMES,
    bootstrap_audio_metrics,
    make_models,
    run_cv,
    tune_threshold,
)
from src.audio.config import EXPECTED_FEATURE_COUNT, AudioConfig  # noqa: E402
from src.audio.dataset import build_dataset, subject_table  # noqa: E402
from src.audio.features import (  # noqa: E402
    META_COLUMNS,
    build_feature_frame,
    extract_vector,
    feature_names,
)
from src.audio.reproducibility import audio_repro_metadata  # noqa: E402
from src.data import (  # noqa: E402
    assert_no_group_leakage,
    cv_splits,
    fit_scaler,
    inner_subject_split,
)

# ----------------------------------------------------------------------------- helpers


def write_wav(path, sr=44100, dur_s=0.6, freq=220.0, channels=1, subtype="PCM_24"):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = sig if channels == 1 else np.stack([sig] * channels, axis=1)
    sf.write(path, data, sr, subtype=subtype)
    return path


@pytest.fixture
def discovery_tree(tmp_path):
    """A raw-dir tree with valid files, the ID22 anomaly, a mismatch, and a no-id file."""
    raw = tmp_path / "raw" / "26-29_09_2017_KCL"
    files = {
        "valid_hc": raw / "ReadText" / "HC" / "ID01_hc_0_0_0.wav",
        "valid_pd": raw / "ReadText" / "PD" / "ID10_pd_0_0_0.wav",
        "id10_again": raw / "SpontaneousDialogue" / "PD" / "ID10_pd_1_0_0.wav",
        "id22_anomaly": raw / "SpontaneousDialogue" / "HC" / "ID22hc_0_0_0.wav",
        "mismatch": raw / "ReadText" / "HC" / "ID03_pd_0_0_0.wav",  # folder HC, name pd
        "no_id": raw / "ReadText" / "HC" / "background_noise.wav",  # no ID<digits>
    }
    for p in files.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")  # discovery reads paths only, not audio
    return tmp_path / "raw", files


def make_fake_frame(rng, n_per_class=10, recs=2, n_feat=12, sep=2.0):
    """A synthetic feature frame (no openSMILE) that is separable by label."""
    rows = []
    sid = 1
    for label in (0, 1):
        for _ in range(n_per_class):
            gk = f"S{sid:02d}"
            for r in range(recs):
                row = {
                    "subject_id": sid,
                    "group_key": gk,
                    "label": label,
                    "cohort_folder": "PD" if label else "HC",
                    "cohort_file": "pd" if label else "hc",
                    "task": "ReadText" if r == 0 else "SpontaneousDialogue",
                    "filename": f"{gk}_{r}.wav",
                    "relpath": f"{gk}_{r}.wav",
                    "duration_s": 1.0,
                    "orig_sample_rate": 44100,
                    "orig_channels": 1,
                }
                feats = rng.normal(label * sep, 1.0, size=n_feat)
                row.update({f"feat_{i:02d}": float(v) for i, v in enumerate(feats)})
                rows.append(row)
            sid += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- discovery


def test_discovers_valid_and_reports_exclusions(discovery_tree):
    raw, _ = discovery_tree
    recs, excl = discovery.discover_recordings(raw)
    # 4 valid: ID01_hc, ID10_pd (x2), ID22hc ; 2 excluded: mismatch, no_id
    assert len(recs) == 4
    reasons = {e.reason for e in excl}
    assert reasons == {"label_folder_filename_mismatch", "no_subject_id"}


def test_tolerant_subject_id_parsing_handles_id22():
    # The whole point of ID(\d+): the missing-underscore ID22 filename still -> 22.
    assert discovery.parse_subject_id("ID22hc_0_0_0.wav") == 22
    assert discovery.parse_subject_id("ID01_hc_0_0_0.wav") == 1
    assert discovery.parse_subject_id("ID10_pd_1_0_0.wav") == 10
    assert discovery.parse_subject_id("background_noise.wav") is None


def test_id22_not_split_into_phantom_subject(discovery_tree):
    raw, _ = discovery_tree
    recs, _ = discovery.discover_recordings(raw)
    by_name = {r.filename: r for r in recs}
    id22 = by_name["ID22hc_0_0_0.wav"]
    assert id22.subject_id == 22
    assert id22.group_key == "S22"
    # a naive split("_")[0] would have produced "ID22hc"; ensure no such phantom key
    assert all("hc" not in r.group_key.lower() for r in recs)


def test_label_derived_from_folder_and_agrees_with_filename(discovery_tree):
    raw, _ = discovery_tree
    recs, _ = discovery.discover_recordings(raw)
    hc = next(r for r in recs if r.subject_id == 1)
    pd_rec = next(r for r in recs if r.subject_id == 10)
    assert (hc.label, hc.cohort_folder, hc.cohort_file) == (0, "HC", "hc")
    assert (pd_rec.label, pd_rec.cohort_folder, pd_rec.cohort_file) == (1, "PD", "pd")


def test_same_subject_two_tasks_share_group_key(discovery_tree):
    raw, _ = discovery_tree
    recs, _ = discovery.discover_recordings(raw)
    id10 = [r for r in recs if r.subject_id == 10]
    assert len(id10) == 2
    assert {r.task for r in id10} == {"ReadText", "SpontaneousDialogue"}
    assert {r.group_key for r in id10} == {"S10"}


# ----------------------------------------------------------------------------- preprocess


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [0.0, 1.0]], dtype=np.float32)  # (2 ch, 2 samples)
    mono = preprocess.to_mono(stereo)
    assert mono.ndim == 1
    np.testing.assert_allclose(mono, [0.5, 2.0])
    assert preprocess.channel_count(stereo) == 2
    assert preprocess.channel_count(mono) == 1


def test_resample_noop_when_rates_match():
    sig = np.linspace(-1, 1, 1000, dtype=np.float32)
    out = preprocess.resample_to(sig, 44100, 44100)
    assert np.array_equal(out, sig)


def test_resample_changes_length_proportionally():
    sig = np.zeros(8000, dtype=np.float32)
    out = preprocess.resample_to(sig, 8000, 44100)
    assert abs(len(out) - 8000 * 44100 / 8000) <= 2


def test_peak_normalize_hits_target_and_is_safe_on_silence():
    sig = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = preprocess.peak_normalize(sig, peak_dbfs=-1.0)
    assert np.max(np.abs(out)) == pytest.approx(10 ** (-1.0 / 20.0), rel=1e-5)
    silent = np.zeros(10, dtype=np.float32)
    assert np.array_equal(preprocess.peak_normalize(silent, -1.0), silent)


def test_preprocess_is_deterministic():
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 0.1, size=44100).astype(np.float32)
    a = preprocess.preprocess_signal(sig, 44100)
    b = preprocess.preprocess_signal(sig, 44100)
    assert np.array_equal(a.signal, b.signal)
    assert a.sample_rate == b.sample_rate == 44100


def test_default_config_is_lossless_at_native_rate():
    # normalization="none", silence="none", 44.1 kHz mono in -> same samples out
    rng = np.random.default_rng(1)
    sig = rng.normal(0, 0.1, size=22050).astype(np.float32)
    out = preprocess.preprocess_signal(sig, 44100)
    assert not out.resampled and not out.normalized
    np.testing.assert_array_equal(out.signal, sig)


def test_preprocess_file_reads_24bit_mono(tmp_path):
    p = write_wav(tmp_path / "a.wav", sr=44100, dur_s=0.6, channels=1)
    out = preprocess.preprocess_file(p)
    assert out.sample_rate == 44100
    assert out.orig_channels == 1
    assert out.duration_s == pytest.approx(0.6, abs=0.01)


def test_preprocess_resamples_nonconforming_file(tmp_path):
    p = write_wav(tmp_path / "b.wav", sr=22050, dur_s=0.6, channels=1)
    out = preprocess.preprocess_file(p)  # default target 44100
    assert out.orig_sample_rate == 22050
    assert out.sample_rate == 44100
    assert out.resampled is True


def test_stereo_file_collapsed_to_mono(tmp_path):
    p = write_wav(tmp_path / "st.wav", sr=44100, dur_s=0.6, channels=2)
    out = preprocess.preprocess_file(p)
    assert out.orig_channels == 2
    assert out.signal.ndim == 1


def test_duration_exclusion_band():
    cfg = AudioConfig()
    short = preprocess.ProcessedAudio(np.zeros(100), 44100, 0.1, 44100, 1, False, False)
    ok = preprocess.ProcessedAudio(np.zeros(44100), 44100, 1.0, 44100, 1, False, False)
    assert "too_short" in preprocess.duration_exclusion_reason(short, cfg)
    assert preprocess.duration_exclusion_reason(ok, cfg) is None


# ----------------------------------------------------------------------------- features


def test_feature_names_count_is_88():
    names = feature_names()
    assert len(names) == EXPECTED_FEATURE_COUNT == 88
    assert len(set(names)) == 88  # unique


def test_extract_vector_shape_and_finite(tmp_path):
    from src.audio.features import build_smile

    p = write_wav(tmp_path / "tone.wav", dur_s=0.7, freq=180)
    processed = preprocess.preprocess_file(p)
    vec = extract_vector(processed, build_smile())
    assert vec.shape == (88,)
    assert np.isfinite(vec).all()


def test_extract_is_deterministic(tmp_path):
    from src.audio.features import build_smile

    p = write_wav(tmp_path / "tone.wav", dur_s=0.7, freq=200)
    processed = preprocess.preprocess_file(p)
    smile = build_smile()
    v1 = extract_vector(processed, smile)
    v2 = extract_vector(processed, smile)
    np.testing.assert_array_equal(v1, v2)


def test_build_feature_frame_schema_and_exclusions(tmp_path):
    raw = tmp_path / "raw"
    write_wav(raw / "ReadText" / "HC" / "ID01_hc_0.wav", dur_s=0.6)
    write_wav(raw / "SpontaneousDialogue" / "HC" / "ID01_hc_1.wav", dur_s=0.6)
    write_wav(raw / "ReadText" / "PD" / "ID02_pd_0.wav", dur_s=0.6)
    write_wav(raw / "SpontaneousDialogue" / "PD" / "ID02_pd_1.wav", dur_s=0.6)
    write_wav(raw / "ReadText" / "HC" / "ID03_hc_0.wav", dur_s=0.2)  # too short -> excluded

    frame, exclusions, names = build_feature_frame(raw_dir=raw)

    assert list(frame.columns) == META_COLUMNS + names
    assert len(names) == 88
    assert len(frame) == 4  # the 0.2s file is excluded
    assert {"S01", "S02"} == set(frame["group_key"])
    # the short file is reported, not silently dropped
    assert any(e["reason"] == "duration" for e in exclusions)
    # both classes present at subject level
    assert set(frame["label"]) == {0, 1}


# ----------------------------------------------------------------------------- dataset / CV


def test_build_dataset_shapes_and_feature_group():
    rng = np.random.default_rng(0)
    frame = make_fake_frame(rng, n_per_class=3, recs=2, n_feat=12)
    data = build_dataset(frame)
    assert data.X.shape == (12, 12)
    assert set(np.unique(data.y)) == {0, 1}
    assert "eGeMAPSv02" in data.feature_groups
    assert len(data.feature_groups["eGeMAPSv02"]) == 12


def test_subject_table_flags_inconsistent_labels():
    rng = np.random.default_rng(0)
    frame = make_fake_frame(rng, n_per_class=2, recs=2, n_feat=4)
    # corrupt one recording's label so a subject has two labels
    frame.loc[0, "label"] = 1 - frame.loc[0, "label"]
    with pytest.raises(ValueError, match="inconsistent label"):
        subject_table(frame)


def test_cv_splits_have_no_subject_leakage_and_cover_all():
    rng = np.random.default_rng(0)
    frame = make_fake_frame(rng, n_per_class=10, recs=2, n_feat=8)
    data = build_dataset(frame)
    seen = np.zeros(len(data.y), dtype=int)
    for train_idx, test_idx in cv_splits(data, n_splits=4, seed=42):
        assert set(data.groups[train_idx]).isdisjoint(set(data.groups[test_idx]))
        assert_no_group_leakage(train_idx, test_idx, data.groups)
        seen[test_idx] += 1
    assert np.all(seen == 1)  # every recording tested exactly once


def test_inner_split_keeps_test_isolated():
    rng = np.random.default_rng(0)
    frame = make_fake_frame(rng, n_per_class=10, recs=2, n_feat=8)
    data = build_dataset(frame)
    train_all, test = cv_splits(data, n_splits=4, seed=42)[0]
    inner_train, val = inner_subject_split(data, train_all, seed=42)
    for a in (inner_train, val):
        assert_no_group_leakage(a, test, data.groups)
    assert_no_group_leakage(inner_train, val, data.groups)


def test_leakage_canary_is_detected():
    rng = np.random.default_rng(0)
    frame = make_fake_frame(rng, n_per_class=4, recs=2, n_feat=4)
    data = build_dataset(frame)
    # force a shared subject across the two index sets
    leaky_train = np.array([0, 1])
    leaky_test = np.array([1, 2])  # index 1 shares subject S01 with train
    with pytest.raises(ValueError, match=r"[Ss]ubject leakage"):
        assert_no_group_leakage(leaky_train, leaky_test, data.groups)


def test_fold_scaler_fits_train_subjects_only():
    rng = np.random.default_rng(0)
    frame = make_fake_frame(rng, n_per_class=10, recs=2, n_feat=8)
    data = build_dataset(frame)
    train_all, _test = cv_splits(data, n_splits=4, seed=42)[0]
    inner_train, _ = inner_subject_split(data, train_all, seed=42)
    scaler = fit_scaler(data.X.iloc[inner_train])
    np.testing.assert_allclose(
        scaler.mean_, data.X.iloc[inner_train].to_numpy().mean(axis=0), rtol=1e-9
    )
    full_mean = data.X.to_numpy().mean(axis=0)
    assert not np.allclose(scaler.mean_, full_mean)


def test_tune_threshold_returns_grid_value_separating_classes():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    thr = tune_threshold(y, prob)
    assert 0.05 <= thr <= 0.95
    pred = (prob >= thr).astype(int)
    np.testing.assert_array_equal(pred, y)


def test_bootstrap_reports_balanced_accuracy_and_brier():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.6, 0.4])
    pred = (prob >= 0.5).astype(int)
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    out = bootstrap_audio_metrics(y, prob, pred, groups, n_boot=100, seed=0)
    assert set(AUDIO_METRIC_NAMES) <= set(out)
    assert "balanced_accuracy" in out and "brier_score" in out
    for stats in out.values():
        assert {"point", "ci_low", "ci_high", "n_boot_valid"} <= set(stats)


def test_run_cv_smoke_produces_subject_metrics():
    rng = np.random.default_rng(0)
    frame = make_fake_frame(rng, n_per_class=10, recs=2, n_feat=12, sep=2.5)
    data = build_dataset(frame)
    results, oof = run_cv(data, n_splits=4, seed=42, n_boot=50)
    assert set(results) == set(make_models(42))
    for res in results.values():
        assert {"recording_level", "subject_level"} <= set(res)
        auc = res["subject_level"]["roc_auc"]["point"]
        assert not np.isnan(auc)
        assert auc > 0.7  # separable synthetic data -> models should rank well
    # OOF predictions filled for every recording
    for name in results:
        assert not np.isnan(oof[name]["prob"]).any()


# ----------------------------------------------------------------------------- config / repro


def test_audio_config_validation_and_roundtrip():
    cfg = AudioConfig()
    d = cfg.as_dict()
    assert d["feature_set"] == "eGeMAPSv02"
    assert d["expected_feature_count"] == 88
    with pytest.raises(ValueError):
        AudioConfig(normalization="loudnorm")
    with pytest.raises(ValueError):
        AudioConfig(min_duration_s=5.0, max_duration_s=1.0)


def test_audio_repro_metadata_shape(tmp_path):
    missing = audio_repro_metadata(seed=7, feature_csv=tmp_path / "nope.csv")
    assert missing["seed"] == 7
    assert missing["feature_set"] == "eGeMAPSv02"
    assert missing["libraries"]["opensmile"] is not None
    assert missing["feature_dataset_sha256"] is None  # file absent
    assert "audio_config" in missing

    real = tmp_path / "feat.csv"
    real.write_text("a,b\n1,2\n", encoding="utf-8")
    present = audio_repro_metadata(seed=7, feature_csv=real)
    assert len(present["feature_dataset_sha256"]) == 64
