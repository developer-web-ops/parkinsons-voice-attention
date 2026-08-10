import numpy as np
import torch

from src.data import class_weights, load_dataset, subject_split
from src.evaluate import compute_metrics
from src.models import GroupAttentionNet, MLPBaseline


def test_dataset_shape_and_groups():
    data = load_dataset()
    assert data.X.shape == (756, 753)
    assert set(np.unique(data.y)) == {0, 1}
    assert len(np.unique(data.groups)) == 252
    assert sum(len(cols) for cols in data.feature_groups.values()) == data.X.shape[1]


def test_subject_split_has_no_leakage():
    data = load_dataset()
    idx = subject_split(data)
    subjects = {name: set(data.groups[i]) for name, i in idx.items()}
    assert not subjects["train"] & subjects["test"]
    assert not subjects["train"] & subjects["val"]
    assert not subjects["val"] & subjects["test"]
    assert sum(len(i) for i in idx.values()) == len(data.y)


def test_class_weights_favour_minority():
    weights = class_weights(np.array([0] * 20 + [1] * 80))
    assert weights[0] > weights[1]


def test_attention_weights_are_a_distribution():
    model = GroupAttentionNet([4, 6, 10]).eval()
    logits, weights = model.forward_with_attention(torch.randn(5, 20))
    assert logits.shape == (5,)
    assert weights.shape == (5, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-5)


def test_mlp_output_shape():
    model = MLPBaseline(20).eval()
    assert model(torch.randn(4, 20)).shape == (4,)


def test_compute_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
