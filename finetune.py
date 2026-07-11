"""Fine-tune a Kaggle-trained RetinaAI EfficientNetB0 model.

Run this script from the RetinaAI project root in Kaggle after the initial
training run has produced ``/kaggle/working/models/retina_model.keras``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.training.data_loader import AptosSequence, load_split
from src.training.evaluation import evaluate_predictions
from src.training.train_model import compute_class_weights


KAGGLE_DATASET_DIR = Path("/kaggle/input/datasets/mariaherrerot/aptos2019")
KAGGLE_MODELS_DIR = Path("/kaggle/working/models")


def find_efficientnet_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Return the nested EfficientNetB0 backbone saved in the classifier model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnetb0" in layer.name.lower():
            return layer
    raise ValueError("EfficientNetB0 backbone was not found in the loaded model.")


def configure_fine_tuning(model: tf.keras.Model, trainable_layers: int = 40) -> None:
    """Unfreeze the final non-BatchNorm backbone layers while freezing all BN."""
    backbone = find_efficientnet_backbone(model)
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    eligible_layers = [
        layer
        for layer in backbone.layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    for layer in eligible_layers[-trainable_layers:]:
        layer.trainable = True

    # Batch normalization layers must stay frozen, including any outside the backbone.
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_for_fine_tuning(model: tf.keras.Model) -> None:
    """Compile with the low learning rate required for fine-tuning."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def fine_tuning_callbacks(models_dir: Path) -> list[tf.keras.callbacks.Callback]:
    """Create fine-tuning convergence and best-model callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            models_dir / "retina_model_finetuned.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_finetuning_artifacts(history: dict[str, list[float]], models_dir: Path) -> None:
    """Save fine-tuning history and the requested accuracy/loss plots."""
    frame = pd.DataFrame(history)
    frame.to_csv(models_dir / "training_history_finetuned.csv", index=False)

    plots = (
        ("accuracy", "val_accuracy", "Accuracy", "accuracy_curve_finetuned.png"),
        ("loss", "val_loss", "Loss", "loss_curve_finetuned.png"),
    )
    for train_key, validation_key, title, filename in plots:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(frame[train_key], label=f"Training {title.lower()}")
        axis.plot(frame[validation_key], label=f"Validation {title.lower()}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(f"Fine-tuning {title}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(models_dir / filename, dpi=150)
        plt.close(figure)


def print_metrics(metrics: dict[str, object]) -> None:
    """Print final test metrics required by the fine-tuning workflow."""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def fine_tune(dataset_dir: Path = KAGGLE_DATASET_DIR, models_dir: Path = KAGGLE_MODELS_DIR) -> None:
    """Fine-tune the saved model using Kaggle's train/validation/test files."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(models_dir / "retina_model.keras")
    configure_fine_tuning(model)
    compile_for_fine_tuning(model)

    train_data = load_split(dataset_dir / "train_1.csv")
    validation_data = load_split(dataset_dir / "valid.csv")
    test_data = load_split(dataset_dir / "test.csv")

    train_sequence = AptosSequence(
        train_data,
        dataset_dir / "train_images" / "train_images",
        batch_size=16,
        augment=True,
        shuffle=True,
    )
    validation_sequence = AptosSequence(
        validation_data,
        dataset_dir / "val_images" / "val_images",
        batch_size=16,
        augment=False,
    )
    test_sequence = AptosSequence(
        test_data,
        dataset_dir / "test_images" / "test_images",
        batch_size=16,
        augment=False,
    )

    class_weights = compute_class_weights(train_data["diagnosis"].to_numpy())
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=15,
        class_weight=class_weights,
        callbacks=fine_tuning_callbacks(models_dir),
    )

    best_model = tf.keras.models.load_model(models_dir / "retina_model_finetuned.keras")
    probabilities = best_model.predict(test_sequence, verbose=1)
    metrics = evaluate_predictions(test_data["diagnosis"].to_numpy(), probabilities)
    save_finetuning_artifacts(history.history, models_dir)
    print_metrics(metrics)


def main() -> None:
    """Parse optional Kaggle paths and start fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune the saved RetinaAI Kaggle model.")
    parser.add_argument("--dataset-dir", type=Path, default=KAGGLE_DATASET_DIR)
    parser.add_argument("--models-dir", type=Path, default=KAGGLE_MODELS_DIR)
    args = parser.parse_args()
    fine_tune(args.dataset_dir, args.models_dir)


if __name__ == "__main__":
    main()
