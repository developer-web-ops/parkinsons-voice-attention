"""Phase 2D orchestrator: compact-CNN comparison against the frozen eGeMAPS baseline.

Reads the recording table from the Phase 2B eGeMAPS feature CSV (metadata only —
same 73 recordings, same row order, so the CNN is evaluated on **identical** subject
folds as the frozen Logistic-Regression baseline) and the raw WAVs under
``data_audio/.../raw``. It then:

* builds the deterministic log-Mel window cache (raw audio -> 16 kHz -> log-Mel);
* runs the canonical single-split (seed 42) CNN evaluation with subject-cluster
  bootstrap CIs, and repeated subject-grouped CV across ``n_repeats`` seeds;
* runs a same-fold *paired* comparison of the CNN against the frozen LR baseline;
* reads (never writes) the frozen Phase 2C LR reference distribution and the
  Phase 2B single-split number for context;
* measures parameter count, model size, training time and CPU inference time;
* writes all artifacts under ``artifacts_audio/cnn/`` and prints an honest verdict.

Consistent with the standing interpretation: the CNN is judged against the frozen
*repeated-CV* LR reference (mean ROC-AUC ~0.76), and counts as an improvement only
if it is *consistently* better across repeated splits — never on a single lucky
split. Nothing here modifies Phase 1, Phase 2A/2B/2C artifacts, or the baseline.

Run with ``python -m src.audio.cnn_phase2d``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.audio.cnn_config import DEFAULT_CNN_CONFIG, CnnConfig
from src.audio.cnn_cv import (
    PAIRED_METRICS,
    RawSignalLoader,
    build_window_cache,
    canonical_bootstrap_cnn,
    fit_cnn_fold,
    paired_cnn_vs_lr,
    repeated_cnn_cv,
)
from src.audio.cnn_model import SmallAudioCNN, count_parameters
from src.audio.dataset import build_dataset, label_balance
from src.audio.features import FEATURE_CSV, load_feature_frame
from src.data import cv_splits, inner_subject_split

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_AUDIO = ROOT / "artifacts_audio"
CNN_DIR = ARTIFACTS_AUDIO / "cnn"
PHASE2C_DIR = ARTIFACTS_AUDIO / "phase2c"
CV_METRICS_JSON = ARTIFACTS_AUDIO / "metrics" / "cv_metrics.json"

# Metric used to decide "consistently better"; higher is better.
DECISION_METRIC = "roc_auc"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_frozen_lr(phase2c_dir: Path, cv_metrics_path: Path) -> dict:
    """Read-only: the frozen LR repeated-CV distribution + single-split number."""
    out: dict = {"available": {}}
    rc = phase2c_dir / "repeated_cv.json"
    if rc.exists():
        lr = json.loads(rc.read_text(encoding="utf-8")).get("logistic_regression")
        if lr:
            out["repeated"] = {"n_repeats": lr.get("n_repeats"), "metrics": lr["metrics"]}
            out["available"]["repeated"] = True
    if Path(cv_metrics_path).exists():
        models = json.loads(Path(cv_metrics_path).read_text(encoding="utf-8")).get("models", {})
        lr = models.get("logistic_regression", {}).get("subject_level")
        if lr:
            out["single_split"] = lr
            out["available"]["single_split"] = True
    return out


def parameter_report(config: CnnConfig) -> dict:
    """Architecture summary + parameter count + float32 model size."""
    model = SmallAudioCNN(config)
    n_params = count_parameters(model)
    per_layer = {
        name: int(p.numel()) for name, p in model.named_parameters() if p.requires_grad
    }
    dummy = torch.zeros(1, 1, config.n_mels, config.n_frames)
    with torch.no_grad():
        out_shape = list(model(dummy).shape)
    return {
        "conv_channels": list(config.conv_channels),
        "kernel_size": config.kernel_size,
        "dropout": config.dropout,
        "input_shape": [1, 1, config.n_mels, config.n_frames],
        "output_shape": out_shape,
        "n_parameters": n_params,
        "float32_size_bytes": n_params * 4,
        "per_parameter_tensor": per_layer,
    }


def _measure_compute(
    cache, data, frame, config: CnnConfig, loader, seed: int
) -> tuple[dict, dict]:
    """Time one-fold training and CPU inference; also return the fitted fold for saving."""
    train_all, test = cv_splits(data, n_splits=config.n_splits, seed=seed)[0]
    inner_train, val = inner_subject_split(data, train_all, seed=seed)

    t0 = time.perf_counter()
    fold = fit_cnn_fold(cache, data, inner_train, val, test, config, seed)
    train_s = time.perf_counter() - t0
    model = fold["model"]
    mean, std = fold["standardizer"]

    # (a) full-pipeline inference (load -> window -> log-Mel -> normalise -> forward).
    from src.audio.spectrogram import MelSpectrogram, extract_windows

    mel = MelSpectrogram(config)
    win = config.window_samples
    sample_relpaths = list(frame["relpath"].iloc[: min(3, len(frame))])
    full_times = []
    for relpath in sample_relpaths:
        t0 = time.perf_counter()
        signal = loader(relpath)
        windows = extract_windows(signal, win, config.max_windows_per_recording)
        tiles = torch.stack([mel(w).unsqueeze(0) for w in windows], dim=0)
        tiles = (tiles - mean[None, None, :, None]) / std[None, None, :, None]
        model.eval()
        with torch.no_grad():
            _ = torch.sigmoid(model(tiles).squeeze(1)).mean().item()
        full_times.append(time.perf_counter() - t0)

    # (b) forward-only latency per window (batch of max_windows).
    batch = torch.zeros(config.max_windows_per_recording, 1, config.n_mels, config.n_frames)
    with torch.no_grad():
        model(batch)  # warm up
        t0 = time.perf_counter()
        for _ in range(20):
            model(batch)
        fwd_s = (time.perf_counter() - t0) / 20.0

    timing = {
        "device": "cpu",
        "torch_threads": int(torch.get_num_threads()),
        "one_fold_train_seconds": round(train_s, 3),
        "epochs_this_fold": fold["epochs"],
        "full_pipeline_infer_seconds_per_recording": round(float(np.mean(full_times)), 4),
        "forward_only_seconds_per_recording_batch": round(fwd_s, 5),
        "forward_only_seconds_per_window": round(fwd_s / config.max_windows_per_recording, 6),
        "note": (
            "CPU-only (no CUDA). Full-pipeline time includes WAV load, resample, "
            "log-Mel and forward pass for one recording."
        ),
    }
    return timing, fold


def _save_example(fold: dict, config: CnnConfig, out_path: Path) -> dict:
    """Persist one trained fold as a diagnostic checkpoint (NOT a deployment artifact)."""
    mean, std = fold["standardizer"]
    payload = {
        "state_dict": fold["model"].state_dict(),
        "config": config.as_dict(),
        "standardizer_mean": mean,
        "standardizer_std": std,
        "note": (
            "Example CNN from fold 0, seed 42 — for size/inspection only. The reported "
            "metrics come from the full subject-grouped CV, not this single fit. ONNX "
            "conversion and deployment are deferred to a later phase (Phase 2D scope)."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {"path": out_path.name, "size_bytes": int(out_path.stat().st_size)}


def _compare(cnn_repeated: dict, cnn_single: dict, frozen_lr: dict, paired: dict) -> dict:
    """CNN vs frozen LR: repeated-CV distributions, single split, and paired summary."""
    rows = {}
    lr_rep = frozen_lr.get("repeated", {}).get("metrics", {})
    for metric in cnn_repeated["metrics"]:
        cnn = cnn_repeated["metrics"][metric]
        row = {
            "cnn_repeated_mean": cnn["mean"],
            "cnn_repeated_band_2_5_97_5": [cnn["p2_5"], cnn["p97_5"]],
        }
        if metric in lr_rep:
            lr = lr_rep[metric]
            row["lr_repeated_mean"] = lr["mean"]
            row["lr_repeated_band_2_5_97_5"] = [lr["p2_5"], lr["p97_5"]]
            row["mean_delta_cnn_minus_lr"] = float(cnn["mean"] - lr["mean"])
            # "consistently better" = CNN's lower band clears the LR mean.
            row["cnn_consistently_better"] = bool(cnn["p2_5"] > lr["mean"])
        rows[metric] = row
    return {
        "reference": "frozen eGeMAPS + Logistic Regression (repeated-CV headline ~0.76 ROC-AUC)",
        "per_metric": rows,
        "paired_same_fold": paired["summary"],
        "cnn_single_split_subject_roc_auc": cnn_single["subject_level"]["roc_auc"],
        "lr_single_split_subject_roc_auc": frozen_lr.get("single_split", {}).get("roc_auc"),
    }


def _verdict(comparison: dict, paired: dict, n_repeats: int) -> dict:
    """Honest, non-overclaiming improvement verdict on the decision metric."""
    row = comparison["per_metric"].get(DECISION_METRIC, {})
    delta = row.get("mean_delta_cnn_minus_lr")
    consistently = row.get("cnn_consistently_better")
    wins = paired["summary"][DECISION_METRIC]["cnn_wins"]
    n = paired["summary"][DECISION_METRIC]["n_seeds"]

    is_improvement = bool(consistently) and wins > n / 2 and (delta or 0) > 0
    if delta is None:
        statement = (
            "Frozen LR repeated-CV reference was unavailable at run time; see the "
            "paired same-fold comparison for the head-to-head result."
        )
    elif is_improvement:
        statement = (
            f"The compact CNN is consistently better than the frozen eGeMAPS+LR baseline "
            f"on {DECISION_METRIC}: mean delta +{delta:.3f}, winning {wins}/{n} paired "
            f"folds, with its repeated-CV band clearing the LR mean."
        )
    else:
        statement = (
            f"The compact CNN does NOT show a consistent improvement over the frozen "
            f"eGeMAPS+LR baseline on {DECISION_METRIC} (mean delta {delta:+.3f}, "
            f"{wins}/{n} paired folds won). The baseline remains the reference; any "
            f"single-split advantage is within the variability of repeated subject-"
            f"grouped splits and is not treated as a real gain."
        )
    return {
        "decision_metric": DECISION_METRIC,
        "n_repeats": n_repeats,
        "mean_delta_cnn_minus_lr": delta,
        "cnn_consistently_better": consistently,
        "paired_folds_won": f"{wins}/{n}",
        "is_improvement": is_improvement,
        "statement": statement,
    }


def run(
    *,
    n_repeats: int = 5,
    n_splits: int = 5,
    base_seed: int = 42,
    seed: int = 42,
    n_boot: int = 2000,
    config: CnnConfig = DEFAULT_CNN_CONFIG,
    feature_csv: Path = FEATURE_CSV,
    phase2c_dir: Path = PHASE2C_DIR,
    cv_metrics_path: Path = CV_METRICS_JSON,
    out_dir: Path = CNN_DIR,
    loader=None,
) -> dict:
    """Execute the full Phase 2D comparison and write artifacts under ``out_dir``."""
    config = CnnConfig(**{**config.as_dict(), "n_splits": n_splits, "n_repeats": n_repeats,
                          "base_seed": base_seed})
    frame = load_feature_frame(feature_csv)
    data = build_dataset(frame)  # X is ignored by the CNN; y/groups drive the folds
    loader = loader or RawSignalLoader(config)

    print(f"[phase2d] {len(frame)} recordings; building log-Mel window cache ...", flush=True)
    t0 = time.perf_counter()
    cache = build_window_cache(frame, config, loader)
    build_s = time.perf_counter() - t0
    print(
        f"[phase2d] cache built in {build_s:.1f}s ({cache.logmel.shape[0]} windows); "
        f"canonical single split (seed {seed}) + bootstrap ...",
        flush=True,
    )

    params = parameter_report(config)
    cnn_single = canonical_bootstrap_cnn(
        cache, data, n_splits=n_splits, seed=seed, n_boot=n_boot, config=config
    )
    print(f"[phase2d] repeated subject-grouped CV ({n_repeats}x{n_splits}) ...", flush=True)
    t0 = time.perf_counter()
    cnn_repeated = repeated_cnn_cv(
        cache, data, n_splits=n_splits, n_repeats=n_repeats, base_seed=base_seed, config=config
    )
    repeated_s = time.perf_counter() - t0
    print(
        f"[phase2d] repeated CV done in {repeated_s:.1f}s; paired same-fold CNN-vs-LR ...",
        flush=True,
    )

    paired = paired_cnn_vs_lr(
        cache, data, n_splits=n_splits, seeds=cnn_repeated["seeds"], config=config
    )
    print("[phase2d] measuring compute (train/infer timing) ...", flush=True)
    timing, fold0 = _measure_compute(cache, data, frame, config, loader, seed)
    timing["window_cache_build_seconds"] = round(build_s, 2)
    timing["repeated_cv_wall_seconds"] = round(repeated_s, 2)
    timing["n_windows_total"] = int(cache.logmel.shape[0])

    frozen_lr = _load_frozen_lr(phase2c_dir, cv_metrics_path)
    comparison = _compare(cnn_repeated, cnn_single, frozen_lr, paired)
    verdict = _verdict(comparison, paired, n_repeats)
    example = _save_example(fold0, config, out_dir / "cnn_example.pt")
    params["saved_example_size_bytes"] = example["size_bytes"]

    metrics_payload = {
        "phase": "2D",
        "model": "compact_cnn",
        "scope": "cnn_vs_frozen_egemaps_baseline_only",
        "config": config.as_dict(),
        "data_balance": label_balance(frame),
        "canonical_single_split": cnn_single,
        "repeated_cv": cnn_repeated,
        "parameters": params,
        "timing": timing,
        "example_checkpoint": example,
    }
    comparison_payload = {
        "phase": "2D",
        "baseline": "frozen eGeMAPS + Logistic Regression (Phase 2B/2C, unmodified)",
        "frozen_lr_reference": frozen_lr,
        "comparison": comparison,
        "paired_same_fold": paired,
        "verdict": verdict,
    }
    summary = {
        "phase": "2D",
        "verdict": verdict,
        "cnn_repeated_roc_auc": cnn_repeated["metrics"]["roc_auc"],
        "cnn_single_split_subject_roc_auc": cnn_single["subject_level"]["roc_auc"],
        "parameters": params["n_parameters"],
        "timing": timing,
        "artifacts": {
            "cnn_config": "cnn_config.json",
            "cnn_metrics": "cnn_metrics.json",
            "cnn_comparison": "cnn_comparison.json",
            "cnn_summary": "cnn_summary.json",
            "example_checkpoint": example["path"],
        },
    }

    print("[phase2d] writing artifacts ...", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "cnn_config.json", config.as_dict())
    _write_json(out_dir / "cnn_metrics.json", metrics_payload)
    _write_json(out_dir / "cnn_comparison.json", comparison_payload)
    _write_json(out_dir / "cnn_summary.json", summary)
    _print_summary(summary, comparison, paired)
    return summary


def _print_summary(summary: dict, comparison: dict, paired: dict) -> None:
    r = summary["cnn_repeated_roc_auc"]
    print("\nPhase 2D — compact CNN vs frozen eGeMAPS+LR baseline")
    print(f"  parameters: {summary['parameters']:,}")
    print(
        f"  CNN repeated ROC-AUC: {r['mean']:.3f}+/-{r['std']:.3f} "
        f"[{r['p2_5']:.3f}, {r['p97_5']:.3f}]  (n={r['n']})"
    )
    lr_row = comparison["per_metric"].get("roc_auc", {})
    if "lr_repeated_mean" in lr_row:
        print(
            f"  LR  repeated ROC-AUC: {lr_row['lr_repeated_mean']:.3f}  "
            f"(delta {lr_row['mean_delta_cnn_minus_lr']:+.3f}, "
            f"consistently_better={lr_row.get('cnn_consistently_better')})"
        )
    for m in PAIRED_METRICS:
        s = paired["summary"][m]
        print(
            f"  paired {m:<18} CNN {s['cnn_mean']:.3f} vs LR {s['lr_mean']:.3f} "
            f"(wins {s['cnn_wins']}/{s['n_seeds']})"
        )
    t = summary["timing"]
    print(
        f"  compute: {t['one_fold_train_seconds']}s/fold train, "
        f"{t['full_pipeline_infer_seconds_per_recording']}s/recording infer (CPU)"
    )
    print(f"  verdict: {summary['verdict']['statement']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    run(
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        base_seed=args.base_seed,
        seed=args.seed,
        n_boot=args.n_boot,
    )
