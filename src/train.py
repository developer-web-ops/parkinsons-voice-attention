"""Train the sklearn baselines, the MLP baseline and the attention network.

Run with ``python -m src.train``. All artifacts (scaler, weights, metrics,
plots) are written to ``artifacts/`` and ``reports/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import evaluate
from src.data import Dataset, class_weights, fit_scaler, load_dataset, subject_split
from src.models import GroupAttentionNet, MLPBaseline

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _loaders(
    X_train: np.ndarray, y_train: np.ndarray, batch_size: int
) -> DataLoader:
    tensors = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=True, drop_last=True)


def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    patience: int = 30,
) -> nn.Module:
    """Train with early stopping on validation ROC-AUC; restores the best weights."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    loader = _loaders(X_train, y_train, batch_size)

    best_auc, best_state, waited = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

        val_auc = evaluate.compute_metrics(y_val, _predict(model, X_val))["roc_auc"]
        if val_auc > best_auc:
            best_auc, waited = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def tune_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> float:
    """Pick the probability cut-off that maximises validation F1."""
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [evaluate.compute_metrics(y_val, prob_val, t)["f1"] for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def subject_level_metrics(
    y: np.ndarray, prob: np.ndarray, groups: np.ndarray, threshold: float = 0.5
) -> dict:
    """Aggregate the three recordings per subject into one averaged prediction."""
    subjects = np.unique(groups)
    y_subj = np.array([y[groups == s][0] for s in subjects])
    p_subj = np.array([prob[groups == s].mean() for s in subjects])
    return evaluate.compute_metrics(y_subj, p_subj, threshold)


def _test_examples(data: Dataset, test_idx: np.ndarray) -> dict[str, dict[str, float]]:
    """One held-out recording per class, so the UI can demo real inputs."""
    examples: dict[str, dict[str, float]] = {}
    for label, name in ((0, "healthy"), (1, "parkinsons")):
        matches = test_idx[data.y[test_idx] == label]
        if len(matches):
            row = data.X.iloc[matches[0]]
            examples[name] = {k: float(v) for k, v in row.items()}
    return examples


def main(epochs: int = 200, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    data: Dataset = load_dataset()
    idx = subject_split(data, seed=seed)
    scaler = fit_scaler(data.X.iloc[idx["train"]])

    splits = {
        name: (scaler.transform(data.X.iloc[i].to_numpy()), data.y[i], data.groups[i])
        for name, i in idx.items()
    }
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, g_test = splits["test"]

    weights = class_weights(y_train)
    pos_weight = float(weights[1] / weights[0])

    results: dict[str, dict] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    sklearn_models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000, C=0.1, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "svm_rbf": SVC(C=1.0, probability=True, class_weight="balanced", random_state=seed),
    }
    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate.compute_metrics(y_test, prob)
        results[name]["subject_level"] = subject_level_metrics(y_test, prob, g_test)
        roc_curves[name] = (y_test, prob)
        joblib.dump(model, ARTIFACTS / f"{name}.joblib")

    mlp = train_torch_model(
        MLPBaseline(X_train.shape[1]), X_train, y_train, X_val, y_val, pos_weight, epochs=epochs
    )
    mlp_threshold = tune_threshold(y_val, _predict(mlp, X_val))
    mlp_prob = _predict(mlp, X_test)
    results["mlp"] = evaluate.compute_metrics(y_test, mlp_prob, mlp_threshold)
    results["mlp"]["subject_level"] = subject_level_metrics(y_test, mlp_prob, g_test, mlp_threshold)
    roc_curves["mlp"] = (y_test, mlp_prob)
    torch.save(mlp.state_dict(), ARTIFACTS / "mlp.pt")

    group_sizes = [len(cols) for cols in data.feature_groups.values()]
    attn = train_torch_model(
        GroupAttentionNet(group_sizes),
        X_train,
        y_train,
        X_val,
        y_val,
        pos_weight,
        epochs=epochs,
        lr=5e-4,
        weight_decay=1e-3,
        patience=40,
    )
    attn_threshold = tune_threshold(y_val, _predict(attn, X_val))
    attn_prob = _predict(attn, X_test)
    results["attention"] = evaluate.compute_metrics(y_test, attn_prob, attn_threshold)
    results["attention"]["subject_level"] = subject_level_metrics(
        y_test, attn_prob, g_test, attn_threshold
    )
    roc_curves["attention"] = (y_test, attn_prob)
    torch.save(attn.state_dict(), ARTIFACTS / "attention.pt")

    attn.eval()
    with torch.no_grad():
        _, attn_weights = attn.forward_with_attention(torch.tensor(X_test, dtype=torch.float32))
    mean_attention = attn_weights.mean(dim=0).numpy()
    group_names = list(data.feature_groups)
    results["attention"]["mean_attention"] = dict(
        zip(group_names, [float(w) for w in mean_attention], strict=True)
    )

    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    metadata = {
        "feature_names": data.feature_names,
        "feature_groups": data.feature_groups,
        "group_sizes": group_sizes,
        "group_names": group_names,
        "feature_medians": data.X.median().to_dict(),
        "feature_min": data.X.min().to_dict(),
        "feature_max": data.X.max().to_dict(),
        "test_subject_ids": [int(s) for s in np.unique(g_test)],
        "thresholds": {"mlp": mlp_threshold, "attention": attn_threshold},
        "examples": _test_examples(data, idx["test"]),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata))
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    evaluate.plot_roc_curves(roc_curves, REPORTS / "roc_curves.png")
    evaluate.plot_attention(group_names, mean_attention, REPORTS / "attention_weights.png")
    for name in ("mlp", "attention"):
        evaluate.plot_confusion_matrix(
            results[name]["confusion_matrix"],
            f"Confusion matrix — {name}",
            REPORTS / f"confusion_matrix_{name}.png",
        )

    for name, res in results.items():
        print(
            f"{name:<20} acc={res['accuracy']:.3f} prec={res['precision']:.3f} "
            f"rec={res['recall']:.3f} f1={res['f1']:.3f} auc={res['roc_auc']:.3f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(epochs=args.epochs, seed=args.seed)
