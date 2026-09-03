"""Fine-tune the leaf classifier on weakly-labeled field leaf crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import config
import preprocess
from datasets.ingest import IMAGE_EXTENSIONS
from field_splits import ADAPT_SPLIT, split_lookup
from train_kfold import compute_class_weights, label_indices


def load_field_manifest_records(
    field_root: Path = config.FIELD_LEAF_DIR,
    split_path: Path | None = None,
) -> list[dict]:
    manifest_path = field_root / "manifest.jsonl"
    if not manifest_path.exists():
        return []

    canopy_splits = split_lookup(split_path=split_path)
    records = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            # The manifest records whichever split was current when the crops were
            # cut. Trusting it would silently reinstate a superseded split, so the
            # split file always wins.
            record["split"] = canopy_splits.get(
                str(Path(record["canopy_path"])), record.get("split", "unknown")
            )
            records.append(record)
    return records


def load_field_arrays(
    field_root: Path = config.FIELD_LEAF_DIR,
    field_split: str = ADAPT_SPLIT,
    split_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    manifest_records = load_field_manifest_records(field_root, split_path=split_path)

    if manifest_records:
        candidate_records = [
            record
            for record in manifest_records
            if field_split == "all" or record.get("split") == field_split
        ]
        for record in candidate_records:
            image_path = Path(record["output_path"])
            class_name = record["unified_class"]
            if not image_path.exists() or class_name not in config.CLASS_TO_INDEX:
                continue
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS)
                images.append(np.asarray(image, dtype=np.float32))
            label = np.zeros(config.NUM_CLASSES, dtype=np.float32)
            label[config.CLASS_TO_INDEX[class_name]] = 1.0
            labels.append(label)
    else:
        for class_name in config.CLASSES:
            class_dir = field_root / class_name
            if not class_dir.exists():
                continue
            class_idx = config.CLASS_TO_INDEX[class_name]
            for image_path in class_dir.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image = image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS)
                    images.append(np.asarray(image, dtype=np.float32))
                label = np.zeros(config.NUM_CLASSES, dtype=np.float32)
                label[class_idx] = 1.0
                labels.append(label)
    return np.asarray(images), np.asarray(labels)


def fine_tune_on_field_leaves(
    base_model_path: Path | None = None,
    field_root: Path = config.FIELD_LEAF_DIR,
    output_path: Path | None = None,
    epochs: int = 10,
    learning_rate: float = 5e-6,
    field_split: str = ADAPT_SPLIT,
    split_path: Path | None = None,
) -> keras.Model:
    if base_model_path is None:
        base_model_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    if output_path is None:
        output_path = config.MODELS_DIR / "autorogue_leaf_classifier_field_adapted.keras"

    model = keras.models.load_model(base_model_path)
    images, labels = load_field_arrays(field_root, field_split=field_split, split_path=split_path)
    if len(images) == 0:
        raise FileNotFoundError(f"No field leaf crops found in {field_root} for split {field_split!r}")
    label_ids = label_indices(labels)
    print(f"Fine-tuning on {len(images)} weak field crops from split {field_split!r}")

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images), seed=config.SEED)
    train_pipeline = preprocess.create_training_pipeline(dataset, augment=True, use_mixup=False)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    model.fit(
        train_pipeline,
        epochs=epochs,
        class_weight=compute_class_weights(label_ids),
    )
    model.save(output_path)
    print(f"Field-adapted classifier saved to {output_path}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune classifier on field leaf crops.")
    parser.add_argument("--base-model-path", type=Path, default=config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME)
    parser.add_argument("--field-root", type=Path, default=config.FIELD_LEAF_DIR)
    parser.add_argument("--field-split", default=ADAPT_SPLIT, help="Field crop split to train on; use 'all' only for experiments.")
    parser.add_argument("--split-path", type=Path, default=config.UNTOUCHED_SPLIT_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=config.MODELS_DIR / "autorogue_leaf_classifier_field_adapted.keras",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    return parser.parse_args()


def main() -> None:
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    args = parse_args()
    fine_tune_on_field_leaves(
        base_model_path=args.base_model_path,
        field_root=args.field_root,
        output_path=args.output_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        field_split=args.field_split,
        split_path=args.split_path,
    )


if __name__ == "__main__":
    main()
