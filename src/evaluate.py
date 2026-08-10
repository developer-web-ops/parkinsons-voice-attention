"""Metric computation and diagnostic plots shared by every model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "n_samples": len(y_true),
    }


def plot_confusion_matrix(cm: list[list[int]], title: str, out_path: Path) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["Healthy", "Parkinson's"])
    ax.set_yticks([0, 1], ["Healthy", "Parkinson's"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] > arr.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (held-out subjects)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attention(group_names: list[str], weights: np.ndarray, out_path: Path) -> None:
    order = np.argsort(weights)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([group_names[i] for i in order], weights[order], color="#4c72b0")
    ax.set_xlabel("Mean attention weight")
    ax.set_title("Attention over acoustic feature groups")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
