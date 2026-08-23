"""Validation-only threshold analysis for the deployed neural nets.

Reproduces the deployed subject split, loads the served models read-only, and
sweeps the decision threshold on the VALIDATION set (never the test set),
reporting operating points under several criteria. The deployed thresholds are
included for comparison. Nothing is retrained or overwritten.

Run with ``python -m src.threshold_analysis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.data import load_dataset, subject_split
from src.models import GroupAttentionNet, MLPBaseline
from src.reproducibility import repro_metadata
from src.stats import threshold_metrics

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def _youden(metrics: dict) -> float:
    sens, spec = metrics["sensitivity"], metrics["specificity"]
    if np.isnan(sens) or np.isnan(spec):
        return float("nan")
    return float(sens + spec - 1.0)


def _sweep(y_val: np.ndarray, prob_val: np.ndarray) -> list[dict]:
    grid = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in grid:
        pred = (prob_val >= t).astype(int)
        row = threshold_metrics(y_val, pred)
        row["threshold"] = float(t)
        row["youden_j"] = _youden(row)
        rows.append(row)
    return rows


def _argbest(rows: list[dict], key: str) -> dict:
    def score(row: dict) -> float:
        value = row[key]
        return -np.inf if np.isnan(value) else value

    return dict(max(rows, key=score))


def _high_sensitivity(rows: list[dict], floor: float = 0.90) -> dict | None:
    eligible = [r for r in rows if not np.isnan(r["sensitivity"]) and r["sensitivity"] >= floor]
    if not eligible:
        return None

    # highest specificity among thresholds that still hit the sensitivity floor
    def spec(row: dict) -> float:
        return -np.inf if np.isnan(row["specificity"]) else row["specificity"]

    return dict(max(eligible, key=spec))


def _model_analysis(name: str, y_val: np.ndarray, prob_val: np.ndarray,
                    deployed_threshold: float) -> dict:
    rows = _sweep(y_val, prob_val)
    deployed = threshold_metrics(y_val, (prob_val >= deployed_threshold).astype(int))
    deployed["threshold"] = float(deployed_threshold)
    deployed["youden_j"] = _youden(deployed)
    return {
        "deployed_threshold": float(deployed_threshold),
        "deployed_threshold_val_metrics": deployed,
        "operating_points": {
            "max_f1": _argbest(rows, "f1"),
            "max_youden_j": _argbest(rows, "youden_j"),
            "high_sensitivity_0.90": _high_sensitivity(rows, 0.90),
        },
        "sweep": rows,
    }


def main(seed: int = 42) -> dict:
    REPORTS.mkdir(exist_ok=True)
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
    scaler = joblib.load(ARTIFACTS / "scaler.joblib")
    feature_names = metadata["feature_names"]
    group_sizes = metadata["group_sizes"]
    thresholds = metadata.get("thresholds", {})

    attention = GroupAttentionNet(group_sizes)
    attention.load_state_dict(torch.load(ARTIFACTS / "attention.pt", map_location="cpu"))
    mlp = MLPBaseline(len(feature_names))
    mlp.load_state_dict(torch.load(ARTIFACTS / "mlp.pt", map_location="cpu"))

    data = load_dataset()
    idx = subject_split(data, seed=seed)
    x_val = scaler.transform(data.X.iloc[idx["val"]].to_numpy())
    y_val = data.y[idx["val"]]

    reproduced_test = sorted(int(s) for s in np.unique(data.groups[idx["test"]]))
    deployed_test = sorted(int(s) for s in metadata.get("test_subject_ids", []))

    report = {
        "config": {
            "analysis": "decision-threshold sweep on the validation split",
            "seed": seed,
            "n_val_recordings": len(y_val),
            "grid": "0.05..0.95 step 0.01 (91 points)",
            "note": (
                "Validation set only; the held-out test set is never used to choose a threshold. "
                "The deployed models are loaded read-only and are not modified."
            ),
        },
        "reproducibility": repro_metadata(seed),
        "split_check": {
            "reproduced_test_matches_deployed": reproduced_test == deployed_test,
            "n_test_subjects_reproduced": len(reproduced_test),
            "n_test_subjects_deployed": len(deployed_test),
        },
        "models": {
            "mlp": _model_analysis("mlp", y_val, _predict(mlp, x_val),
                                   float(thresholds.get("mlp", 0.5))),
            "attention": _model_analysis("attention", y_val, _predict(attention, x_val),
                                         float(thresholds.get("attention", 0.5))),
        },
    }
    (REPORTS / "threshold_analysis.json").write_text(json.dumps(report, indent=2))

    matches = report["split_check"]["reproduced_test_matches_deployed"]
    print(f"Split reproduced matches deployed: {matches}")
    for name, res in report["models"].items():
        op = res["operating_points"]
        mf1 = op["max_f1"]
        print(
            f"{name:<10} deployed_thr={res['deployed_threshold']:.2f}  "
            f"max-F1 thr={mf1['threshold']:.2f} (F1={mf1['f1']:.3f}, "
            f"sens={mf1['sensitivity']:.3f}, spec={mf1['specificity']:.3f})"
        )
    print("Wrote reports/threshold_analysis.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
