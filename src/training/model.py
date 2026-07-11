"""EfficientNetB0 classifier architecture for RetinaAI."""

from __future__ import annotations

import tensorflow as tf


NUM_CLASSES = 5
INPUT_SHAPE = (224, 224, 3)


def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 0.001,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Create a frozen ImageNet-pretrained EfficientNetB0 classifier.

    Inputs are preprocessed in the data pipeline with EfficientNet's official
    ``preprocess_input`` function before they reach the model.
    """
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="retinal_image")
    features = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="diagnosis")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="retina_efficientnetb0")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model, backbone
