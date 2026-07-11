"""End-to-end training orchestration for the RetinaAI classifier."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.training.data_loader import AptosSequence, load_split
from src.training.evaluation import evaluate_predictions, save_history_and_plots
from src.training.model import build_model


KAGGLE_DATASET_DIR = Path("/kaggle/input/datasets/mariaherrerot/aptos2019")
KAGGLE_WORKING_MODELS_DIR = Path("/kaggle/working/models")


@dataclass(frozen=True)
class TrainingPaths:
    """CSV, image, and output locations for a training environment."""

    train_csv: Path
    validation_csv: Path
    evaluation_csv: Path
    train_images_dir: Path
    validation_images_dir: Path
    evaluation_images_dir: Path
    models_dir: Path


def running_on_kaggle() -> bool:
    """Return whether the program is executing inside a Kaggle notebook runtime."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def resolve_training_paths(project_root: Path | None = None) -> TrainingPaths:
    """Resolve Kaggle input paths or the existing local project paths.

    Kaggle's supplied train/validation/test manifests are used directly; no
    local ``dataset/processed`` split manifests are read in that environment.
    """
    if running_on_kaggle():
        return TrainingPaths(
            train_csv=KAGGLE_DATASET_DIR / "train_1.csv",
            validation_csv=KAGGLE_DATASET_DIR / "valid.csv",
            evaluation_csv=KAGGLE_DATASET_DIR / "test.csv",
            train_images_dir=KAGGLE_DATASET_DIR / "train_images" / "train_images",
            validation_images_dir=KAGGLE_DATASET_DIR / "val_images" / "val_images",
            evaluation_images_dir=KAGGLE_DATASET_DIR / "test_images" / "test_images",
            models_dir=KAGGLE_WORKING_MODELS_DIR,
        )

    root = Path.cwd() if project_root is None else Path(project_root)
    processed_dir = root / "dataset" / "processed"
    images_dir = root / "dataset" / "raw" / "train_images" / "train_images"
    return TrainingPaths(
        train_csv=processed_dir / "train_split.csv",
        validation_csv=processed_dir / "val_split.csv",
        evaluation_csv=processed_dir / "test_split.csv",
        train_images_dir=images_dir,
        validation_images_dir=images_dir,
        evaluation_images_dir=images_dir,
        models_dir=root / "models",
    )


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
    paths: TrainingPaths,
    batch_size: int = 16,
    epochs: int = 30,
) -> dict[str, object]:
    """Train, evaluate, and save an EfficientNetB0 diabetic-retinopathy model."""
    train_data = load_split(paths.train_csv)
    validation_data = load_split(paths.validation_csv)
    evaluation_data = load_split(paths.evaluation_csv)

    train_sequence = AptosSequence(train_data, paths.train_images_dir, batch_size, augment=True, shuffle=True)
    validation_sequence = AptosSequence(validation_data, paths.validation_images_dir, batch_size, augment=False)
    evaluation_sequence = AptosSequence(evaluation_data, paths.evaluation_images_dir, batch_size, augment=False)

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_model()
    class_weights = compute_class_weights(train_data["diagnosis"].to_numpy())
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=training_callbacks(paths.models_dir),
    )

    # Reload the saved best checkpoint before validation evaluation.
    model_path = paths.models_dir / "retina_model.keras"
    print("=" * 60)
    print(f"Expected model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"ModelCheckpoint did not produce the expected file: {model_path}"
        )
    print(f"Model size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 60)

    best_model = tf.keras.models.load_model(model_path)
    probabilities = best_model.predict(evaluation_sequence, verbose=1)
    metrics = evaluate_predictions(evaluation_data["diagnosis"].to_numpy(), probabilities)
    save_history_and_plots(history.history, paths.models_dir)
    final_model_path = paths.models_dir / "retina_model_final.keras"
    best_model.save(final_model_path)

    print("Best checkpoint saved successfully.")
    print(model_path.resolve())
    print("Final model saved successfully.")
    print(final_model_path.resolve())
    print("Files in models directory:")
    for file in sorted(paths.models_dir.iterdir()):
        print(file.resolve())
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    paths = resolve_training_paths()
    environment = "Kaggle" if running_on_kaggle() else "local"
    print(f"Training environment: {environment}")
    print(f"Model output directory: {paths.models_dir}")
    metrics = train(paths, batch_size=args.batch_size, epochs=args.epochs)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
