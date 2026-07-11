"""Keras data loading for APTOS CSV manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing.preprocess import RetinaImagePreprocessor


class AptosSequence(tf.keras.utils.Sequence):
    """Load CSV-referenced retinal images in batches without duplicating files."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: Path,
        batch_size: int = 16,
        augment: bool = False,
        shuffle: bool = False,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.preprocessor = RetinaImagePreprocessor(images_dir)
        self.indices = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        rows = self.indices[start : start + self.batch_size]
        batch = self.dataframe.iloc[rows]
        images = np.stack(
            [self.preprocessor.load_image(row.id_code, augment=self.augment) for row in batch.itertuples()]
        )
        # The shared preprocessor returns RGB floats in 0–1. EfficientNetB0
        # expects the ImageNet input convention used by its pretrained model.
        images = tf.keras.applications.efficientnet.preprocess_input(images * 255.0)
        labels = tf.keras.utils.to_categorical(batch["diagnosis"].astype(int), self.num_classes)
        return images.astype(np.float32), labels.astype(np.float32)

    def on_epoch_end(self) -> None:
        """Shuffle only the training sequence between epochs."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_split(path: Path) -> pd.DataFrame:
    """Read and validate an APTOS split CSV manifest."""
    dataframe = pd.read_csv(path)
    required = {"id_code", "diagnosis"}
    if not required.issubset(dataframe.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return dataframe
