"""Grad-CAM generation for RetinaAI classifier predictions."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Locate the nested EfficientNetB0 model in the trained classifier."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnetb0" in layer.name.lower():
            return layer
    raise ValueError("EfficientNetB0 backbone was not found in the loaded model.")


def _find_last_convolutional_layer(backbone: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the final convolutional feature layer in EfficientNetB0."""
    convolutional_layers = [
        layer
        for layer in backbone.layers
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
    ]
    if not convolutional_layers:
        raise ValueError("No convolutional layer was found in EfficientNetB0.")
    return convolutional_layers[-1]


def _classifier_layers(model: tf.keras.Model, backbone: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Return the unchanged classification head layers after the backbone."""
    return list(model.layers[model.layers.index(backbone) + 1 :])


def generate_gradcam(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    predicted_class: int,
    source_image_path: Path,
    output_dir: Path,
) -> Path:
    """Generate and save a normalized Grad-CAM overlay for one prediction."""
    backbone = _find_backbone(model)
    last_conv_layer = _find_last_convolutional_layer(backbone)
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_layer.output, backbone.output],
    )

    with tf.GradientTape() as tape:
        convolution_output, features = backbone_grad_model(image_batch, training=False)
        predictions = features
        for layer in _classifier_layers(model, backbone):
            predictions = layer(predictions, training=False)
        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(class_score, convolution_output)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(convolution_output[0] * weights, axis=-1)
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
    return output_path
