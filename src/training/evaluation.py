"""Evaluation metrics and plots for RetinaAI training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Calculate classification metrics and a five-class confusion matrix."""
    y_pred = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))),
    }


def save_history_and_plots(history: dict[str, list[float]], models_dir: Path) -> None:
    """Save epoch history plus accuracy and loss figures."""
    models_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(models_dir / "training_history.csv", index=False)

    plots = (("accuracy", "val_accuracy", "Accuracy", "accuracy_curve.png"), ("loss", "val_loss", "Loss", "loss_curve.png"))
    for train_key, validation_key, label, filename in plots:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history_frame[train_key], label=f"Training {label.lower()}")
        axis.plot(history_frame[validation_key], label=f"Validation {label.lower()}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.set_title(f"Training and Validation {label}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(models_dir / filename, dpi=150)
        plt.close(figure)
