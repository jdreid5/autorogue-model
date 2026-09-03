"""Train the leaf segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import config
from field_splits import excluded_canopy_stems
from segment.data import (
    SegmentationPair,
    dataset_from_pairs,
    list_combined_segmentation_pairs,
    list_segmentation_pairs,
    split_segmentation_pairs,
)
from segment.model import compile_segmenter, create_segmenter


def source_counts(pairs: list[tuple[str, str] | SegmentationPair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        source = pair.source if isinstance(pair, SegmentationPair) else "source"
        counts[source] = counts.get(source, 0) + 1
    return counts


def train_segmenter(
    image_dir: Path = config.SEGMENTATION_IMAGE_DIR,
    mask_dir: Path = config.SEGMENTATION_MASK_DIR,
    field_image_dir: Path | None = config.FIELD_SEGMENTATION_IMAGE_DIR,
    field_mask_dir: Path | None = config.FIELD_SEGMENTATION_MASK_DIR,
    output_path: Path | None = None,
    val_fraction: float = 0.2,
    initial_model_path: Path | None = None,
    learning_rate: float | None = None,
    augment: bool = True,
    field_sample_weight: float = config.SEGMENTATION_FIELD_SAMPLE_WEIGHT,
    split_path: Path | None = None,
) -> keras.Model:
    if output_path is None:
        output_path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME

    held_out_canopies: list[str] = []
    excluded_field_pairs: list[str] = []
    if field_image_dir is None or field_mask_dir is None:
        pairs = list_segmentation_pairs(image_dir=image_dir, mask_dir=mask_dir)
    else:
        held_out_canopies = sorted(excluded_canopy_stems(split_path=split_path))
        pairs, excluded_field_pairs = list_combined_segmentation_pairs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            field_image_dir=field_image_dir,
            field_mask_dir=field_mask_dir,
            field_sample_weight=field_sample_weight,
            excluded_canopy_stems=held_out_canopies,
        )
        print(
            f"Held-out canopies known to the split: {len(held_out_canopies)}; "
            f"field masks dropped from training: {len(excluded_field_pairs)}"
        )
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found in {image_dir} and {mask_dir}; "
            f"field dirs checked: {field_image_dir}, {field_mask_dir}"
        )
    train_pairs, val_pairs = split_segmentation_pairs(pairs, val_fraction=val_fraction)
    train_dataset = dataset_from_pairs(train_pairs, shuffle=True, augment=augment)
    val_dataset = dataset_from_pairs(val_pairs, shuffle=False, augment=False) if val_pairs else None

    if learning_rate is None:
        learning_rate = config.SEGMENTATION_FINETUNE_LR if initial_model_path else config.SEGMENTATION_LR
    if initial_model_path:
        model = keras.models.load_model(initial_model_path, compile=False)
    else:
        model = create_segmenter()
    model = compile_segmenter(model, learning_rate=learning_rate)
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
                "train_sources": source_counts(train_pairs),
                "validation_sources": source_counts(val_pairs),
                "field_sample_weight": field_sample_weight,
                "split_path": str(split_path or config.UNTOUCHED_SPLIT_PATH),
                "n_held_out_canopies": len(held_out_canopies),
                "excluded_field_pairs": excluded_field_pairs,
                "initial_model_path": str(initial_model_path) if initial_model_path else None,
                "learning_rate": float(learning_rate),
                "augment": augment,
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
    parser.add_argument("--field-image-dir", type=Path, default=config.FIELD_SEGMENTATION_IMAGE_DIR)
    parser.add_argument("--field-mask-dir", type=Path, default=config.FIELD_SEGMENTATION_MASK_DIR)
    parser.add_argument("--output-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--initial-model-path", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--field-sample-weight", type=float, default=config.SEGMENTATION_FIELD_SAMPLE_WEIGHT)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=config.UNTOUCHED_SPLIT_PATH,
        help="Field split whose held-out canopies must not enter segmenter training.",
    )
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    args = parse_args()
    train_segmenter(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        field_image_dir=args.field_image_dir,
        field_mask_dir=args.field_mask_dir,
        output_path=args.output_path,
        val_fraction=args.val_fraction,
        initial_model_path=args.initial_model_path,
        learning_rate=args.learning_rate,
        augment=not args.no_augment,
        field_sample_weight=args.field_sample_weight,
        split_path=args.split_path,
    )


if __name__ == "__main__":
    main()
