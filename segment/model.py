"""Lightweight TensorFlow leaf segmentation model."""

from __future__ import annotations

import keras
from keras import layers

import config


def conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def create_segmenter(input_shape=config.SEGMENTATION_IMG_SHAPE) -> keras.Model:
    """Create a compact U-Net style semantic leaf segmenter."""
    inputs = keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    bridge = conv_block(p3, 256)

    u3 = layers.UpSampling2D()(bridge)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 128)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 64)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="leaf_mask")(c6)
    return keras.Model(inputs, outputs, name="autorogue_leaf_segmenter")


def dice_coefficient(y_true, y_pred, smooth: float = 1.0):
    y_true_f = keras.ops.reshape(y_true, [-1])
    y_pred_f = keras.ops.reshape(y_pred, [-1])
    intersection = keras.ops.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        keras.ops.sum(y_true_f) + keras.ops.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def binary_iou(y_true, y_pred, smooth: float = 1.0):
    y_true_f = keras.ops.reshape(y_true > 0.5, [-1])
    y_pred_f = keras.ops.reshape(y_pred > 0.5, [-1])
    y_true_f = keras.ops.cast(y_true_f, "float32")
    y_pred_f = keras.ops.cast(y_pred_f, "float32")
    intersection = keras.ops.sum(y_true_f * y_pred_f)
    union = keras.ops.sum(y_true_f) + keras.ops.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def compile_segmenter(model: keras.Model, learning_rate: float = config.SEGMENTATION_LR) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=dice_loss,
        metrics=[dice_coefficient, binary_iou, keras.metrics.BinaryAccuracy(name="mask_accuracy")],
    )
    return model
