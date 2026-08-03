# preprocess.py
"""
Unified image preprocessing and augmentation for multi-class leaf classification.
"""

from pathlib import Path
from typing import Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import layers

import config

# Maps [0, 255] pixels onto the [-1, 1] range MobileNetV3 expects.
MOBILENET_INPUT_SCALE = 1.0 / 127.5
MOBILENET_INPUT_OFFSET = -1.0

# Geometric augmentation exposes areas outside the source image. Keras defaults to
# "reflect", which mirrors leaf tissue into regions that are flat neutral background
# at inference time. Fill with that background instead. Applies to [0, 255] inputs,
# so augmentation must run before preprocess_for_mobilenet.
GEOMETRIC_FILL = {
    "fill_mode": "constant",
    "fill_value": float(config.NEUTRAL_BACKGROUND_VALUE),
}


def load_dataset(
    directory: Path,
    subset: Optional[str] = None,
    validation_split: Optional[float] = None,
    shuffle: bool = True,
    seed: int = config.SEED
) -> tf.data.Dataset:
    """
    Load images from directory structure into tf.data.Dataset.
    
    Args:
        directory: Path to data directory with class subdirectories
        subset: "training" or "validation" if using validation_split
        validation_split: Fraction for validation (e.g., 0.2)
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
    
    Returns:
        tf.data.Dataset with (image, label) pairs
    """
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,
        color_mode="rgb",
        batch_size=None,  # Return unbatched for more flexibility
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
    )


def get_augmentation_layer(training: bool = True) -> keras.Sequential:
    """
    Create augmentation layer for training or inference.
    
    Args:
        training: If True, returns training augmentations; else minimal augmentation
    
    Returns:
        Keras Sequential model with augmentation layers
    """
    if training:
        aug_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(config.ROTATION_RANGE / 360, **GEOMETRIC_FILL),  # Convert degrees to fraction
            layers.RandomZoom(config.ZOOM_RANGE, **GEOMETRIC_FILL),
            layers.RandomBrightness(config.BRIGHTNESS_RANGE),
            layers.RandomContrast(config.CONTRAST_RANGE),
            # Slight translation for position invariance
            layers.RandomTranslation(0.1, 0.1, **GEOMETRIC_FILL),
        ]
        # Add vertical flip if enabled
        if config.VERTICAL_FLIP:
            aug_layers.insert(1, layers.RandomFlip("vertical"))
        return keras.Sequential(aug_layers, name="augmentation")
    else:
        # Minimal augmentation for TTA
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05, **GEOMETRIC_FILL),
        ], name="tta_augmentation")


def preprocess_for_mobilenet(images):
    """
    Preprocess images for MobileNetV3.
    MobileNetV3 expects inputs in [-1, 1] range.

    The backbone is built with `include_preprocessing=False`, so its internal
    Rescaling layer is absent. `keras.applications.mobilenet_v3.preprocess_input`
    cannot be used here because it is a no-op kept only for API compatibility;
    the scaling has to be applied explicitly.

    Args:
        images: NumPy array or tensor of images in [0, 255] range

    Returns:
        Preprocessed images in [-1, 1] range, matching the input container type
    """
    if isinstance(images, np.ndarray):
        return images.astype(np.float32) * MOBILENET_INPUT_SCALE + MOBILENET_INPUT_OFFSET
    return tf.cast(images, tf.float32) * MOBILENET_INPUT_SCALE + MOBILENET_INPUT_OFFSET


def mixup(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = config.MIXUP_ALPHA
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply MixUp augmentation: blend pairs of images and their labels.
    
    Args:
        images: Batch of images
        labels: Batch of labels
        alpha: Beta distribution parameter (higher = more mixing)
    
    Returns:
        Mixed images and labels
    """
    batch_size = tf.shape(images)[0]
    
    # Sample mixing coefficient from Beta distribution
    lam = tf.random.uniform([], 0, 1)
    if alpha > 0:
        lam = tfp_beta_sample(alpha, alpha) if alpha > 0 else lam
    
    # Create shuffled indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images and labels
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels


def tfp_beta_sample(alpha: float, beta: float) -> tf.Tensor:
    """Sample from Beta distribution using gamma distributions."""
    gamma_1 = tf.random.gamma([], alpha)
    gamma_2 = tf.random.gamma([], beta)
    return gamma_1 / (gamma_1 + gamma_2)


def cutmix(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = config.CUTMIX_ALPHA
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply CutMix augmentation: cut and paste patches between images.
    
    Args:
        images: Batch of images
        labels: Batch of labels  
        alpha: Beta distribution parameter for box size
    
    Returns:
        CutMix augmented images and labels
    """
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]
    
    # Sample lambda from Beta distribution
    lam = tfp_beta_sample(alpha, alpha)
    
    # Calculate cut box size
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
    
    # Random box center
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    
    # Box boundaries (clipped to image bounds)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)
    
    # Create mask
    mask = tf.ones((img_h, img_w), dtype=tf.float32)
    padding = [[y1, img_h - y2], [x1, img_w - x2]]
    cut_region = tf.zeros((y2 - y1, x2 - x1), dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.reshape(tf.stack(tf.meshgrid(
            tf.range(y1, y2), tf.range(x1, x2), indexing='ij'
        ), axis=-1), [-1, 2]),
        tf.zeros([(y2 - y1) * (x2 - x1)], dtype=tf.float32)
    )
    mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)
    
    # Shuffle indices for pairing
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    # Apply CutMix
    mixed_images = images * mask + shuffled_images * (1 - mask)
    
    # Adjust lambda based on actual cut area
    actual_lam = 1 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
    mixed_labels = actual_lam * labels + (1 - actual_lam) * shuffled_labels
    
    return mixed_images, mixed_labels


def apply_mix_augmentation(
    images: tf.Tensor,
    labels: tf.Tensor,
    prob: float = config.MIX_PROB
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly apply either MixUp or CutMix with given probability.
    
    Args:
        images: Batch of images
        labels: Batch of labels
        prob: Probability of applying any mix augmentation
    
    Returns:
        Augmented images and labels
    """
    if prob <= 0 or (config.MIXUP_ALPHA <= 0 and config.CUTMIX_ALPHA <= 0):
        return images, labels

    # Decide whether to apply mix augmentation
    apply_mix = tf.random.uniform([]) < prob

    def apply_selected_mix():
        if config.MIXUP_ALPHA <= 0:
            return cutmix(images, labels)
        if config.CUTMIX_ALPHA <= 0:
            return mixup(images, labels)
        use_mixup = tf.random.uniform([]) < 0.5
        return tf.cond(
            use_mixup,
            lambda: mixup(images, labels),
            lambda: cutmix(images, labels),
        )

    return tf.cond(
        apply_mix,
        apply_selected_mix,
        lambda: (images, labels),
    )


def create_training_pipeline(
    dataset: tf.data.Dataset,
    augment: bool = True,
    use_mixup: bool = True
) -> tf.data.Dataset:
    """
    Create the full training data pipeline with augmentation.
    
    Args:
        dataset: Raw tf.data.Dataset
        augment: Whether to apply augmentation
        use_mixup: Whether to apply MixUp/CutMix
    
    Returns:
        Prepared tf.data.Dataset for training
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Batch the dataset
    dataset = dataset.batch(config.BATCH_SIZE)
    
    if augment:
        aug_layer = get_augmentation_layer(training=True)
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Apply MixUp/CutMix after batching
    if use_mixup:
        dataset = dataset.map(
            lambda x, y: apply_mix_augmentation(x, y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Preprocess for MobileNetV3
    dataset = dataset.map(
        lambda x, y: (preprocess_for_mobilenet(x), y),
        num_parallel_calls=AUTOTUNE
    )
    
    # Prefetch for performance
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


def create_validation_pipeline(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Create validation/test data pipeline (no augmentation).
    
    Args:
        dataset: Raw tf.data.Dataset
    
    Returns:
        Prepared tf.data.Dataset for validation/testing
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.map(
        lambda x, y: (preprocess_for_mobilenet(x), y),
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


def test_time_augmentation(
    model: keras.Model,
    images: tf.Tensor,
    n_augments: int = config.TTA_AUGMENTS
) -> tf.Tensor:
    """
    Apply test-time augmentation and average predictions.
    
    Args:
        model: Trained model
        images: Batch of images in [0, 255] range
        n_augments: Number of augmented predictions to average
    
    Returns:
        Averaged predictions
    """
    tta_aug = get_augmentation_layer(training=False)
    predictions = []
    
    # Original prediction
    predictions.append(model(preprocess_for_mobilenet(images), training=False))
    
    # Augmented predictions. Augment before preprocessing so the constant fill
    # value stays in the same [0, 255] scale as the pixels.
    for _ in range(n_augments - 1):
        aug_images = tta_aug(images, training=True)
        predictions.append(model(preprocess_for_mobilenet(aug_images), training=False))
    
    # Average all class-probability predictions.
    return tf.reduce_mean(tf.stack(predictions, axis=0), axis=0)


def load_all_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load train, validation, and test datasets from split directories.
    
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    train_ds = load_dataset(config.TRAIN_DIR, shuffle=True)
    val_ds = load_dataset(config.VAL_DIR, shuffle=False)
    test_ds = load_dataset(config.TEST_DIR, shuffle=False)
    
    return train_ds, val_ds, test_ds


def load_for_kfold() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all training+validation data for k-fold cross-validation.
    Returns numpy arrays for use with sklearn KFold.
    
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    # Combine train and validation for k-fold
    train_ds = load_dataset(config.TRAIN_DIR, shuffle=False)
    val_ds = load_dataset(config.VAL_DIR, shuffle=False)
    
    # Convert to numpy
    train_images = []
    train_labels = []
    
    for img, label in train_ds:
        train_images.append(img.numpy())
        train_labels.append(label.numpy())
    
    for img, label in val_ds:
        train_images.append(img.numpy())
        train_labels.append(label.numpy())
    
    return np.array(train_images), np.array(train_labels)


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data as numpy arrays.
    
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    test_ds = load_dataset(config.TEST_DIR, shuffle=False)
    
    images = []
    labels = []
    
    for img, label in test_ds:
        images.append(img.numpy())
        labels.append(label.numpy())
    
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    # Quick test of the pipeline
    print("Testing preprocessing pipeline...")
    
    train_ds, val_ds, test_ds = load_all_data()
    
    print(f"Train samples: {len(list(train_ds))}")
    print(f"Val samples: {len(list(val_ds))}")
    print(f"Test samples: {len(list(test_ds))}")
    
    # Test pipeline
    train_pipeline = create_training_pipeline(train_ds)
    
    for batch_images, batch_labels in train_pipeline.take(1):
        print(f"Batch shape: {batch_images.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Image range: [{batch_images.numpy().min():.2f}, {batch_images.numpy().max():.2f}]")
    
    print("Pipeline test complete!")

