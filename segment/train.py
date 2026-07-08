"""Train the leaf segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import config
from segment.data import dataset_from_pairs, list_segmentation_pairs, split_segmentation_pairs
from segment.model import compile_segmenter, create_segmenter


def train_segmenter(
    image_dir: Path = config.SEGMENTATION_IMAGE_DIR,
    mask_dir: Path = config.SEGMENTATION_MASK_DIR,
    output_path: Path | None = None,
    val_fraction: float = 0.2,
) -> keras.Model:
    if output_path is None:
        output_path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME

    pairs = list_segmentation_pairs(image_dir=image_dir, mask_dir=mask_dir)
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found in {image_dir} and {mask_dir}")
    train_pairs, val_pairs = split_segmentation_pairs(pairs, val_fraction=val_fraction)
    train_dataset = dataset_from_pairs(train_pairs, shuffle=True)
    val_dataset = dataset_from_pairs(val_pairs, shuffle=False) if val_pairs else None
    model = compile_segmenter(create_segmenter())
    monitor = "val_dice_coefficient" if val_dataset is not None else "dice_coefficient"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor=monitor,
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.SEGMENTATION_EPOCHS,
        callbacks=callbacks,
    )
    model.save(output_path)
    history_path = config.OUTPUTS_DIR / "segmenter_training_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_train": len(train_pairs),
                "n_validation": len(val_pairs),
                "monitor": monitor,
                "history": {key: [float(value) for value in values] for key, values in history.history.items()},
            },
            handle,
            indent=2,
        )
    print(f"Segmenter saved to {output_path}")
    print(f"Training history saved to {history_path}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Autorogue leaf segmenter.")
    parser.add_argument("--image-dir", type=Path, default=config.SEGMENTATION_IMAGE_DIR)
    parser.add_argument("--mask-dir", type=Path, default=config.SEGMENTATION_MASK_DIR)
    parser.add_argument("--output-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    args = parse_args()
    train_segmenter(args.image_dir, args.mask_dir, args.output_path, args.val_fraction)


if __name__ == "__main__":
    main()
