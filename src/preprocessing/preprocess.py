"""APTOS 2019 image preprocessing and stratified dataset splitting.

This module never writes image files. It keeps the original APTOS images in
``dataset/raw`` and writes only CSV manifests to ``dataset/processed``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)
RANDOM_STATE: Final[int] = 42


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem locations for the APTOS dataset and output split manifests."""

    raw_dir: Path
    processed_dir: Path

    @property
    def labels_csv(self) -> Path:
        return self.raw_dir / "train_1.csv"

    @property
    def images_dir(self) -> Path:
        return self.raw_dir / "train_images" / "train_images"


class RetinaImagePreprocessor:
    """Load and transform retinal images in memory for model input."""

    def __init__(self, images_dir: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> None:
        self.images_dir = Path(images_dir)
        self.image_size = image_size

    def image_path(self, id_code: str) -> Path:
        """Return the expected PNG path for an APTOS image identifier."""
        return self.images_dir / f"{id_code}.png"

    def load_image(self, id_code: str, augment: bool = False) -> np.ndarray:
        """Load, resize, RGB-convert, normalize, and optionally augment an image.

        Augmentation is intended exclusively for the training data loader. Use
        ``augment=False`` for validation, test, and prediction data.
        """
        path = self.image_path(id_code)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        return self.augment(image) if augment else image

    @staticmethod
    def augment(image: np.ndarray) -> np.ndarray:
        """Apply randomized training-only flip, rotation, zoom, and brightness."""
        height, width = image.shape[:2]

        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)

        angle = float(np.random.uniform(-15.0, 15.0))
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(
            image,
            rotation,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        zoom = float(np.random.uniform(0.9, 1.1))
        zoom_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, zoom)
        image = cv2.warpAffine(
            image,
            zoom_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        brightness = float(np.random.uniform(0.8, 1.2))
        return np.clip(image * brightness, 0.0, 1.0).astype(np.float32)


def validate_labels(labels: pd.DataFrame, images_dir: Path) -> None:
    """Validate the input CSV schema and that its referenced images exist."""
    expected_columns = {"id_code", "diagnosis"}
    if not expected_columns.issubset(labels.columns):
        raise ValueError(f"CSV must contain {sorted(expected_columns)}; got {list(labels.columns)}")
    if labels[["id_code", "diagnosis"]].isna().any().any():
        raise ValueError("CSV contains missing id_code or diagnosis values.")

    available_ids = {path.stem for path in images_dir.glob("*.png")}
    missing = sorted(set(labels["id_code"]) - available_ids)
    if missing:
        example = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} CSV images are missing; examples: {example}")


def create_stratified_splits(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic 80/10/10 train/validation/test stratified splits."""
    train, holdout = train_test_split(
        labels,
        test_size=0.20,
        stratify=labels["diagnosis"],
        random_state=RANDOM_STATE,
    )
    validation, test = train_test_split(
        holdout,
        test_size=0.50,
        stratify=holdout["diagnosis"],
        random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(paths: DatasetPaths) -> dict[str, pd.DataFrame]:
    """Read labels, validate them, and save only CSV split manifests."""
    labels = pd.read_csv(paths.labels_csv)
    validate_labels(labels, paths.images_dir)
    train, validation, test = create_stratified_splits(labels)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": validation, "test": test}
    filenames = {"train": "train_split.csv", "val": "val_split.csv", "test": "test_split.csv"}
    for name, split in splits.items():
        split.to_csv(paths.processed_dir / filenames[name], index=False)
    return splits


def split_summary(splits: dict[str, pd.DataFrame]) -> str:
    """Format split counts and per-class distribution for console output."""
    lines = ["APTOS 2019 split summary"]
    for name in ("train", "val", "test"):
        split = splits[name]
        distribution = split["diagnosis"].value_counts().sort_index()
        classes = ", ".join(f"{label}: {count}" for label, count in distribution.items())
        lines.append(f"{name.title()}: {len(split)} images | diagnosis counts: {classes}")
    return "\n".join(lines)


def main() -> None:
    """Create the three CSV manifests from command-line arguments."""
    parser = argparse.ArgumentParser(description="Create stratified APTOS CSV splits.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    args = parser.parse_args()

    splits = save_splits(DatasetPaths(args.raw_dir, args.processed_dir))
    print(split_summary(splits))


if __name__ == "__main__":
    main()
