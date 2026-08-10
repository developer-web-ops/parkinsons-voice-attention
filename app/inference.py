"""Model loading and single-sample inference with attention + SHAP explanations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch

from src.models import GroupAttentionNet, LogitWrapper, MLPBaseline

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

MODEL_NAMES = ("attention", "mlp")


class ModelBundle:
    """Holds the scaler, both PyTorch models and the metadata needed to score."""

    def __init__(self) -> None:
        self.metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
        self.feature_names: list[str] = self.metadata["feature_names"]
        self.group_names: list[str] = self.metadata["group_names"]
        self.feature_groups: dict[str, list[str]] = self.metadata["feature_groups"]
        self.thresholds: dict[str, float] = self.metadata.get("thresholds", {})
        self.scaler = joblib.load(ARTIFACTS / "scaler.joblib")

        self.attention = GroupAttentionNet(self.metadata["group_sizes"])
        self.attention.load_state_dict(torch.load(ARTIFACTS / "attention.pt", map_location="cpu"))
        self.attention.eval()

        self.mlp = MLPBaseline(len(self.feature_names))
        self.mlp.load_state_dict(torch.load(ARTIFACTS / "mlp.pt", map_location="cpu"))
        self.mlp.eval()

        background_path = ARTIFACTS / "shap_background.npy"
        self.background = (
            np.load(background_path).astype(np.float32) if background_path.exists() else None
        )
        self._explainer = None

    def threshold(self, model: str) -> float:
        return float(self.thresholds.get(model, 0.5))

    def defaults(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.metadata["feature_medians"].items()}

    def vectorise(self, features: dict[str, float]) -> np.ndarray:
        """Build a full feature vector, filling anything unspecified with the median."""
        medians = self.metadata["feature_medians"]
        values = [float(features.get(name, medians[name])) for name in self.feature_names]
        return np.asarray(values, dtype=np.float64).reshape(1, -1)

    def scale(self, raw: np.ndarray) -> np.ndarray:
        return self.scaler.transform(raw).astype(np.float32)

    def predict(self, raw: np.ndarray, model: str = "attention") -> dict:
        x = torch.tensor(self.scale(raw))
        if model == "mlp":
            with torch.no_grad():
                prob = float(torch.sigmoid(self.mlp(x))[0])
            attention = None
        else:
            with torch.no_grad():
                logit, weights = self.attention.forward_with_attention(x)
            prob = float(torch.sigmoid(logit)[0])
            pairs = zip(self.group_names, weights[0].tolist(), strict=True)
            attention = {g: float(w) for g, w in pairs}

        threshold = self.threshold(model)
        return {
            "model": model,
            "probability": prob,
            "prediction": int(prob >= threshold),
            "label": "Parkinson's" if prob >= threshold else "Healthy",
            "threshold": threshold,
            "attention": attention,
        }

    def shap_top_features(self, raw: np.ndarray, top_k: int = 12) -> list[dict]:
        """Per-request SHAP attributions for the attention model."""
        if self.background is None:
            return []
        import shap  # imported lazily: it is only needed when explanations are requested

        if self._explainer is None:
            self._explainer = shap.GradientExplainer(
                LogitWrapper(self.attention), torch.tensor(self.background)
            )
        values = np.asarray(self._explainer.shap_values(torch.tensor(self.scale(raw))))
        if values.ndim == 3:
            values = values[..., 0]
        values = values[0]
        order = np.argsort(np.abs(values))[::-1][:top_k]
        return [
            {"feature": self.feature_names[i], "shap_value": float(values[i])} for i in order
        ]


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return ModelBundle()


def load_report(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}
