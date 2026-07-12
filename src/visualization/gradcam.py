"""Grad-CAM generation for RetinaAI classifier predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnetb0" in layer.name.lower():
            return layer
    raise ValueError("EfficientNetB0 backbone was not found in the loaded model.")


def _find_last_convolutional_layer(backbone: tf.keras.Model) -> tf.keras.layers.Layer:
    convolutional_layers = [
        layer
        for layer in backbone.layers
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
    ]
    if not convolutional_layers:
        raise ValueError("No convolutional layer was found in EfficientNetB0.")
    return convolutional_layers[-1]


def generate_gradcam(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    predicted_class: int,
    source_image_path: Path,
    output_dir: Path,
) -> Path:
    backbone = _find_backbone(model)
    last_conv_layer = _find_last_convolutional_layer(backbone)

    backbone_idx = model.layers.index(backbone)
    classifier_layers = model.layers[backbone_idx + 1:]

    backbone_input = backbone.input
    if isinstance(backbone_input, (list, tuple)):
        backbone_input = backbone_input[0]

    backbone_grad_model = tf.keras.Model(
        inputs=backbone_input,
        outputs=[last_conv_layer.output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_output, backbone_features = backbone_grad_model(image_batch, training=False)
        predictions = backbone_features
        for layer in classifier_layers:
            predictions = layer(predictions, training=False)
        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(class_score, conv_output)
    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated. "
            "The gradient tape could not trace from the classifier output "
            "back to the last convolutional layer. Verify that "
            "backbone.input resolves to a valid tensor and that "
            "classifier_layers connect to backbone.output."
        )

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if float(maximum) == 0:
        raise RuntimeError("Grad-CAM produced an empty heatmap.")
    normalized_heatmap = (heatmap / maximum).numpy()

    original = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("The uploaded image could not be read for Grad-CAM overlay.")
    resized_heatmap = cv2.resize(normalized_heatmap, (original.shape[1], original.shape[0]))
    colored_heatmap = cv2.applyColorMap(np.uint8(resized_heatmap * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.58, colored_heatmap, 0.42, 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_image_path.stem}_gradcam_{uuid4().hex[:8]}.png"
    if not cv2.imwrite(str(output_path), overlay):
        raise IOError(f"Could not save Grad-CAM overlay: {output_path}")

    logger.info("Saved Grad-CAM overlay to %s", output_path)
    return output_path
