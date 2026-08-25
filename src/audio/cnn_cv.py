"""Subject-grouped cross-validation for the compact audio CNN (Phase 2D).

This reuses — verbatim — the Phase 2B/2C evaluation machinery so the CNN is scored
under *exactly* the same protocol as the frozen eGeMAPS + Logistic-Regression
baseline, and can be compared to it on **identical folds**:

* folds come from :func:`src.data.cv_splits` on the same recording table (same row
  order as the eGeMAPS feature frame) -> the CNN and LR see the same subject
  partition for a given seed;
* the inner subject-grouped validation split, the leakage guards, the max-F1
  threshold, Platt calibration, the augmented metric set and the subject-cluster
  bootstrap are the same functions the baseline uses;
* no subject appears in both train and validation/test (asserted every fold);
* the log-Mel per-bin standardisation is fit on inner-training windows only.

Per outer fold:

    windows(inner-train) -> per-Mel-bin mean/std           (training windows only)
    train SmallAudioCNN with class-weighted BCE + early stopping on inner-val loss
    window logits -> sigmoid -> mean over each recording's windows
    Platt-scale recording probs on the inner-validation fold (never sees test)
    threshold = argmax-F1 on calibrated inner-validation recording probs
    predict calibrated recording probs on the held-out test fold

Recording-level out-of-fold predictions are pooled and aggregated to one prediction
per subject (mean), exactly like the baseline. Writes nothing; the orchestrator
:mod:`src.audio.cnn_phase2d` persists artifacts under ``artifacts_audio/cnn/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from src.audio.baseline_cv import (
    AUDIO_METRIC_NAMES,
    _augmented_metrics,
    bootstrap_audio_metrics,
    tune_threshold,
)
from src.audio.cnn_config import DEFAULT_CNN_CONFIG, CnnConfig
from src.audio.cnn_model import SmallAudioCNN
from src.audio.robustness import REPEAT_METRIC_NAMES, _distribution, _subject_pool
from src.audio.spectrogram import MelSpectrogram, extract_windows
from src.data import (
    Dataset,
    assert_no_group_leakage,
    cv_splits,
    inner_subject_split,
)

# Metrics reported in the paired same-fold comparison; brier is lower-is-better.
PAIRED_METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier_score")
HIGHER_IS_BETTER = {
    "roc_auc": True,
    "pr_auc": True,
    "balanced_accuracy": True,
    "f1": True,
    "brier_score": False,
}


# --------------------------------------------------------------------------- cache


@dataclass
class WindowCache:
    """Pre-computed *raw* (unnormalised) log-Mel windows for every recording.

    Spectrograms do not depend on the fold, so they are computed once and reused
    across folds/repeats; only the per-bin standardisation is refit per fold.
    """

    logmel: torch.Tensor  # (n_windows, 1, n_mels, n_frames), float32, raw
    y_win: np.ndarray  # (n_windows,) window label = its recording's label
    rec_of_win: np.ndarray  # (n_windows,) recording (row) index
    n_recordings: int
    config: CnnConfig


def build_window_cache(
    frame: pd.DataFrame, config: CnnConfig, loader
) -> WindowCache:
    """Compute log-Mel windows for every recording in ``frame`` (row order kept).

    ``loader(relpath) -> 1-D float32 signal at ``config.sample_rate`` decouples this
    from disk so tests can supply synthetic signals. Recording ``i`` is row ``i`` of
    ``frame``; that is also row ``i`` of the :class:`Dataset` used for the folds, so
    window -> recording -> subject mapping stays exact.
    """
    mel = MelSpectrogram(config)
    win = config.window_samples
    tiles: list[torch.Tensor] = []
    y_win: list[int] = []
    rec_of_win: list[int] = []
    for i, row in enumerate(frame.itertuples(index=False)):
        signal = loader(row.relpath)
        windows = extract_windows(signal, win, config.max_windows_per_recording)
        for w in windows:
            tiles.append(mel(w).unsqueeze(0))  # (1, n_mels, n_frames)
            y_win.append(int(row.label))
            rec_of_win.append(i)
    logmel = torch.stack(tiles, dim=0)  # (n_windows, 1, n_mels, n_frames)
    return WindowCache(
        logmel=logmel,
        y_win=np.asarray(y_win, dtype=np.int64),
        rec_of_win=np.asarray(rec_of_win, dtype=np.int64),
        n_recordings=len(frame),
        config=config,
    )


class RawSignalLoader:
    """Default loader: read ``raw_dir/relpath`` and preprocess to ``sample_rate``.

    Reuses the deterministic :func:`src.audio.preprocess.preprocess_file` path
    (mono + polyphase resample, no normalisation/silence handling), so the CNN's
    16 kHz input is produced by the same audited code as the baseline's I/O.
    """

    def __init__(self, config: CnnConfig, raw_dir=None) -> None:
        from src.audio.config import AudioConfig
        from src.audio.features import RAW_DIR

        self.raw_dir = raw_dir or RAW_DIR
        self.audio_config = AudioConfig(
            target_sample_rate=config.sample_rate,
            mono=config.mono,
            normalization="none",
            silence="none",
            min_duration_s=config.min_duration_s,
            max_duration_s=config.max_duration_s,
        )

    def __call__(self, relpath: str) -> np.ndarray:
        from src.audio.preprocess import preprocess_file

        processed = preprocess_file(self.raw_dir / relpath, self.audio_config)
        return processed.signal


# --------------------------------------------------------------------------- train


def _take(x: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    return x[torch.as_tensor(idx, dtype=torch.long)]


def _fit_standardizer(x_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-Mel-bin mean/std over inner-training windows (dims batch, channel, time)."""
    mean = x_train.mean(dim=(0, 1, 3))
    std = x_train.std(dim=(0, 1, 3)) + 1e-6
    return mean, std


def _apply_standardizer(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (x - mean[None, None, :, None]) / std[None, None, :, None]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def train_cnn(
    x_tr: torch.Tensor,
    y_tr: np.ndarray,
    x_val: torch.Tensor,
    y_val: np.ndarray,
    config: CnnConfig,
    seed: int,
) -> tuple[SmallAudioCNN, int]:
    """Train with class-weighted BCE and early stopping on inner-val loss.

    Deterministic given ``seed``: weight init via ``torch.manual_seed`` and batch
    shuffling via a seeded NumPy generator. Restores the best-val weights.
    """
    torch.manual_seed(seed)
    model = SmallAudioCNN(config)

    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = (
        torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
        if config.class_weighted_loss
        else None
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    rng = np.random.default_rng(seed)
    n = x_tr.shape[0]

    best_loss = float("inf")
    best_state: dict | None = None
    patience = 0
    epochs_run = 0
    for _epoch in range(config.max_epochs):
        epochs_run += 1
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val_t).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epochs_run


def _window_probs(model: SmallAudioCNN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1)).numpy()


def _aggregate_to_recordings(
    win_probs: np.ndarray, win_recs: np.ndarray, recs: np.ndarray
) -> np.ndarray:
    """Mean window probability per recording, in the order of ``recs``."""
    return np.array([win_probs[win_recs == r].mean() for r in recs])


def fit_cnn_fold(
    cache: WindowCache,
    data: Dataset,
    inner_train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    config: CnnConfig,
    seed: int,
) -> dict:
    """Train + calibrate + threshold on one fold; return test recording predictions."""
    tr_w = np.flatnonzero(np.isin(cache.rec_of_win, inner_train))
    va_w = np.flatnonzero(np.isin(cache.rec_of_win, val))
    te_w = np.flatnonzero(np.isin(cache.rec_of_win, test))

    mean, std = _fit_standardizer(_take(cache.logmel, tr_w))  # training windows only
    x_tr = _apply_standardizer(_take(cache.logmel, tr_w), mean, std)
    x_val = _apply_standardizer(_take(cache.logmel, va_w), mean, std)
    x_test = _apply_standardizer(_take(cache.logmel, te_w), mean, std)

    model, epochs = train_cnn(x_tr, cache.y_win[tr_w], x_val, cache.y_win[va_w], config, seed)

    val_rec_prob = _aggregate_to_recordings(
        _window_probs(model, x_val), cache.rec_of_win[va_w], val
    )
    test_rec_prob = _aggregate_to_recordings(
        _window_probs(model, x_test), cache.rec_of_win[te_w], test
    )
    val_rec_y = data.y[val]

    if len(np.unique(val_rec_y)) >= 2:
        platt = LogisticRegression()
        platt.fit(_logit(val_rec_prob).reshape(-1, 1), val_rec_y)
        val_cal = platt.predict_proba(_logit(val_rec_prob).reshape(-1, 1))[:, 1]
        test_cal = platt.predict_proba(_logit(test_rec_prob).reshape(-1, 1))[:, 1]
        was_cal = True
    else:  # degenerate single-class validation fold: skip calibration
        val_cal, test_cal, was_cal = val_rec_prob, test_rec_prob, False

    thr = tune_threshold(val_rec_y, val_cal)
    return {
        "test_prob": test_cal,
        "threshold": float(thr),
        "calibrated": bool(was_cal),
        "epochs": int(epochs),
        "model": model,
        "standardizer": (mean, std),
    }


def _fold_predictions_cnn(
    cache: WindowCache, data: Dataset, *, n_splits: int, seed: int, config: CnnConfig
):
    """Pooled out-of-fold recording prob / threshold for the CNN at one seed."""
    n = cache.n_recordings
    prob = np.full(n, np.nan)
    thr = np.full(n, np.nan)
    fold_thresholds: list[float] = []
    calibrated: list[bool] = []
    epochs: list[int] = []
    for train_all, test in cv_splits(data, n_splits=n_splits, seed=seed):
        assert_no_group_leakage(train_all, test, data.groups)
        inner_train, val = inner_subject_split(data, train_all, seed=seed)
        assert_no_group_leakage(inner_train, test, data.groups)
        assert_no_group_leakage(val, test, data.groups)
        assert_no_group_leakage(inner_train, val, data.groups)

        fold = fit_cnn_fold(cache, data, inner_train, val, test, config, seed)
        prob[test] = fold["test_prob"]
        thr[test] = fold["threshold"]
        fold_thresholds.append(fold["threshold"])
        calibrated.append(fold["calibrated"])
        epochs.append(fold["epochs"])
    return prob, thr, fold_thresholds, calibrated, epochs


def reference_oof_cnn(
    cache: WindowCache,
    data: Dataset,
    *,
    n_splits: int = 5,
    seed: int = 42,
    config: CnnConfig = DEFAULT_CNN_CONFIG,
):
    """Canonical (seed=42) out-of-fold ``(prob, pred, thr)`` per recording + diagnostics."""
    prob, thr, fold_thresholds, calibrated, epochs = _fold_predictions_cnn(
        cache, data, n_splits=n_splits, seed=seed, config=config
    )
    pred = (prob >= thr).astype(int)
    return prob, pred, thr, fold_thresholds, calibrated, epochs


def _one_repeat_cnn(
    cache: WindowCache, data: Dataset, *, n_splits: int, seed: int, config: CnnConfig
):
    prob, thr, fold_thresholds, _cal, _ep = _fold_predictions_cnn(
        cache, data, n_splits=n_splits, seed=seed, config=config
    )
    subj_labels, p_subj, pred_subj = _subject_pool(data, prob, thr)
    return _augmented_metrics(subj_labels, p_subj, pred_subj), fold_thresholds


def repeated_cnn_cv(
    cache: WindowCache,
    data: Dataset,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
    config: CnnConfig = DEFAULT_CNN_CONFIG,
) -> dict:
    """Repeat subject-grouped CV for the CNN across ``n_repeats`` re-shuffled splits.

    Same contract as :func:`src.audio.robustness.repeated_grouped_cv` but for the
    single CNN model, so the two distributions are directly comparable. Repeat ``r``
    uses ``base_seed + r`` for the split, inner split and model init.
    """
    seeds = [base_seed + r for r in range(n_repeats)]
    per_repeat = {m: [] for m in REPEAT_METRIC_NAMES}
    all_thresholds: list[float] = []
    for seed in seeds:
        metrics, fold_thresholds = _one_repeat_cnn(
            cache, data, n_splits=n_splits, seed=seed, config=config
        )
        for m in REPEAT_METRIC_NAMES:
            per_repeat[m].append(metrics[m])
        all_thresholds.extend(fold_thresholds)
    return {
        "model": "compact_cnn",
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "base_seed": base_seed,
        "seeds": seeds,
        "metrics": {m: _distribution(per_repeat[m]) for m in REPEAT_METRIC_NAMES},
        "per_repeat": {m: [float(v) for v in per_repeat[m]] for m in REPEAT_METRIC_NAMES},
        "threshold_stability": _distribution(all_thresholds, thresholds=True),
        "fold_thresholds": [float(t) for t in all_thresholds],
    }


def canonical_bootstrap_cnn(
    cache: WindowCache,
    data: Dataset,
    *,
    n_splits: int = 5,
    seed: int = 42,
    n_boot: int = 2000,
    config: CnnConfig = DEFAULT_CNN_CONFIG,
) -> dict:
    """Single-split (seed=42) subject- and recording-level metrics with bootstrap CIs.

    Mirrors the baseline's single-split view so the CNN has the *same* two
    uncertainty lenses (single-split bootstrap CI + repeated-CV band). This is a
    favorable-vs-honest distinction the user requires kept explicit.
    """
    prob, pred, thr, fold_thresholds, calibrated, epochs = reference_oof_cnn(
        cache, data, n_splits=n_splits, seed=seed, config=config
    )
    subj_labels, p_subj, pred_subj = _subject_pool(data, prob, thr)
    return {
        "seed": seed,
        "recording_level": bootstrap_audio_metrics(
            data.y, prob, pred, data.groups, n_boot=n_boot, seed=seed
        ),
        "subject_level": bootstrap_audio_metrics(
            subj_labels, p_subj, pred_subj, np.unique(data.groups), n_boot=n_boot, seed=seed
        ),
        "fold_thresholds": [float(t) for t in fold_thresholds],
        "folds_calibrated": [bool(c) for c in calibrated],
        "epochs_per_fold": [int(e) for e in epochs],
        "metric_names": list(AUDIO_METRIC_NAMES),
    }


def paired_cnn_vs_lr(
    cache: WindowCache,
    data: Dataset,
    *,
    n_splits: int = 5,
    seeds: list[int],
    config: CnnConfig = DEFAULT_CNN_CONFIG,
) -> dict:
    """Same-fold paired comparison: CNN vs frozen LR baseline across seeds.

    For each seed both models are evaluated on the *identical* subject partition
    (:func:`cv_splits` is a pure function of labels+groups+seed), so per-seed deltas
    are paired. ``lr`` is the frozen eGeMAPS + Logistic-Regression baseline evaluated
    via :func:`src.audio.robustness._one_repeat` on the same ``data``.
    """
    from src.audio.robustness import _one_repeat

    per_seed: list[dict] = []
    for seed in seeds:
        cnn_m, _ = _one_repeat_cnn(cache, data, n_splits=n_splits, seed=seed, config=config)
        lr_m, _ = _one_repeat(data, "logistic_regression", n_splits=n_splits, seed=seed)
        per_seed.append(
            {
                "seed": seed,
                "cnn": {k: float(cnn_m[k]) for k in PAIRED_METRICS},
                "lr": {k: float(lr_m[k]) for k in PAIRED_METRICS},
            }
        )

    summary = {}
    for k in PAIRED_METRICS:
        cnn_vals = np.array([s["cnn"][k] for s in per_seed], dtype=float)
        lr_vals = np.array([s["lr"][k] for s in per_seed], dtype=float)
        deltas = cnn_vals - lr_vals
        # "win" means CNN is better, respecting metric direction.
        wins = int(np.sum(deltas > 0) if HIGHER_IS_BETTER[k] else np.sum(deltas < 0))
        summary[k] = {
            "higher_is_better": HIGHER_IS_BETTER[k],
            "cnn_mean": float(np.nanmean(cnn_vals)),
            "lr_mean": float(np.nanmean(lr_vals)),
            "mean_delta_cnn_minus_lr": float(np.nanmean(deltas)),
            "cnn_wins": wins,
            "n_seeds": len(per_seed),
        }
    return {
        "seeds": list(seeds),
        "metrics": list(PAIRED_METRICS),
        "per_seed": per_seed,
        "summary": summary,
        "note": (
            "Paired on identical subject folds per seed. brier_score is "
            "lower-is-better; 'cnn_wins' already accounts for metric direction."
        ),
    }
