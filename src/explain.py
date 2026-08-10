"""Explainability: attention weights over feature groups and SHAP attributions.

Run with ``python -m src.explain`` after training. Produces
``reports/shap_top_features.png`` and ``artifacts/shap_background.npy`` (the
background sample the API reuses at inference time).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.data import load_dataset, subject_split
from src.evaluate import plot_attention
from src.models import GroupAttentionNet, LogitWrapper

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 60


def load_attention_model(group_sizes: list[int]) -> GroupAttentionNet:
    model = GroupAttentionNet(group_sizes)
    model.load_state_dict(torch.load(ARTIFACTS / "attention.pt", map_location="cpu"))
    model.eval()
    return model


def group_shap(shap_values: np.ndarray, feature_groups: dict[str, list[str]]) -> dict[str, float]:
    """Sum |SHAP| within each acoustic block so it is comparable to attention."""
    totals: dict[str, float] = {}
    start = 0
    for group, cols in feature_groups.items():
        end = start + len(cols)
        totals[group] = float(np.abs(shap_values[:, start:end]).sum(axis=1).mean())
        start = end
    return totals


def main() -> dict:
    data = load_dataset()
    idx = subject_split(data)
    scaler = joblib.load(ARTIFACTS / "scaler.joblib")
    group_sizes = [len(cols) for cols in data.feature_groups.values()]
    model = load_attention_model(group_sizes)

    X_train = scaler.transform(data.X.iloc[idx["train"]].to_numpy()).astype(np.float32)
    X_test = scaler.transform(data.X.iloc[idx["test"]].to_numpy()).astype(np.float32)

    rng = np.random.default_rng(42)
    chosen = rng.choice(len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False)
    background = X_train[chosen]
    sample = X_test[: EXPLAIN_SIZE]

    explainer = shap.GradientExplainer(LogitWrapper(model), torch.tensor(background))
    shap_values = np.asarray(explainer.shap_values(torch.tensor(sample)))
    if shap_values.ndim == 3:  # (n, features, outputs)
        shap_values = shap_values[..., 0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    names = data.feature_names

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order][::-1], mean_abs[order][::-1], color="#dd8452")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 features driving the attention network")
    fig.tight_layout()
    fig.savefig(REPORTS / "shap_top_features.png", dpi=150)
    plt.close(fig)

    with torch.no_grad():
        _, weights = model.forward_with_attention(torch.tensor(X_test))
    mean_attention = weights.mean(dim=0).numpy()
    plot_attention(list(data.feature_groups), mean_attention, REPORTS / "attention_weights.png")

    summary = {
        "top_features": [
            {"feature": names[i], "mean_abs_shap": float(mean_abs[i])} for i in order
        ],
        "group_shap": group_shap(shap_values, data.feature_groups),
        "mean_attention": {
            g: float(w) for g, w in zip(data.feature_groups, mean_attention, strict=True)
        },
    }
    (REPORTS / "explainability.json").write_text(json.dumps(summary, indent=2))
    np.save(ARTIFACTS / "shap_background.npy", background[:50])

    print("Top features:", ", ".join(f["feature"] for f in summary["top_features"][:8]))
    print("Attention:", {k: round(v, 3) for k, v in summary["mean_attention"].items()})
    return summary


if __name__ == "__main__":
    main()
