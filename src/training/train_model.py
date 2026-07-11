"""End-to-end training orchestration for the RetinaAI classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.training.data_loader import AptosSequence, load_split
from src.training.evaluation import evaluate_predictions, save_history_and_plots
from src.training.model import build_model


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights from integer diagnosis labels."""
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights, strict=True)}


def training_callbacks(models_dir: Path) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for best-model saving and convergence control."""
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=models_dir / "retina_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


def train(
    processed_dir: Path,
    images_dir: Path,
    models_dir: Path,
    batch_size: int = 16,
    epochs: int = 30,
) -> dict[str, object]:
    """Train, evaluate, and save an EfficientNetB0 diabetic-retinopathy model."""
    train_data = load_split(processed_dir / "train_split.csv")
    validation_data = load_split(processed_dir / "val_split.csv")

    train_sequence = AptosSequence(train_data, images_dir, batch_size, augment=True, shuffle=True)
    validation_sequence = AptosSequence(validation_data, images_dir, batch_size, augment=False)

    models_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_model()
    class_weights = compute_class_weights(train_data["diagnosis"].to_numpy())
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=training_callbacks(models_dir),
    )

    # Reload the saved best checkpoint before validation evaluation.
    best_model = tf.keras.models.load_model(models_dir / "retina_model.keras")
    probabilities = best_model.predict(validation_sequence, verbose=1)
    metrics = evaluate_predictions(validation_data["diagnosis"].to_numpy(), probabilities)
    save_history_and_plots(history.history, models_dir)
    return metrics


def print_metrics(metrics: dict[str, object]) -> None:
    """Print required validation metrics in a readable form."""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def main() -> None:
    """Train from the standard RetinaAI project layout."""
    parser = argparse.ArgumentParser(description="Train the RetinaAI EfficientNetB0 classifier.")
    parser.add_argument("--processed-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/raw/train_images/train_images"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    metrics = train(
        args.processed_dir,
        args.images_dir,
        args.models_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
