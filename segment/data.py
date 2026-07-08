"""
Segmentation dataset preparation.

Supports common polygon annotation formats used by Zenodo/LabelMe and COCO-style
Roboflow exports. Outputs paired RGB images and binary masks for training the
semantic leaf segmenter; post-processing splits semantic masks into instances.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps

import config
from datasets.ingest import IMAGE_EXTENSIONS


def rasterize_polygons(size: tuple[int, int], polygons: Iterable[list[list[float]]]) -> Image.Image:
    """Rasterize polygons into a binary mask."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(points, outline=255, fill=255)
    return mask


def load_labelme_polygons(annotation_path: Path) -> list[list[list[float]]]:
    """Read LabelMe-like JSON with shapes[].points."""
    with open(annotation_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    polygons = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if points:
            polygons.append(points)
    return polygons


def load_json_allowing_truncated_labelme_image_data(annotation_path: Path) -> dict:
    """Read JSON, recovering LabelMe files truncated in the unused imageData field."""
    try:
        with open(annotation_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        text = annotation_path.read_text(encoding="utf-8")
        image_data_index = text.find('\n  "imageData"')
        if image_data_index == -1:
            raise

        prefix = text[:image_data_index].rstrip()
        if not prefix.endswith(","):
            raise

        return json.loads(prefix[:-1] + "\n}")


def load_generic_polygon_json(annotation_path: Path) -> list[list[list[float]]]:
    """Read common polygon JSON structures from phenotyping exports."""
    data = load_json_allowing_truncated_labelme_image_data(annotation_path)

    if "shapes" in data:
        polygons = []
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if points:
                polygons.append(points)
        return polygons

    polygons = []
    candidates = data.get("polygons") or data.get("annotations") or data.get("regions") or []
    for item in candidates:
        if isinstance(item, dict):
            points = item.get("points") or item.get("polygon") or item.get("segmentation")
        else:
            points = item
        if not points:
            continue
        if points and isinstance(points[0], (int, float)):
            flat = points
            points = [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]
        polygons.append(points)
    return polygons


def convert_polygon_pair(image_path: Path, annotation_path: Path, output_image: Path, output_mask: Path) -> None:
    """Convert one image/json polygon pair into normalized image + mask files."""
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        polygons = load_generic_polygon_json(annotation_path)
        mask = rasterize_polygons(image.size, polygons)
        image = ImageOps.contain(image, (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE))
        mask = ImageOps.contain(mask, (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE))

        image_canvas = Image.new("RGB", (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE), (0, 0, 0))
        mask_canvas = Image.new("L", (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE), 0)
        image_canvas.paste(image, ((config.SEGMENTATION_IMG_SIZE - image.width) // 2, (config.SEGMENTATION_IMG_SIZE - image.height) // 2))
        mask_canvas.paste(mask, ((config.SEGMENTATION_IMG_SIZE - mask.width) // 2, (config.SEGMENTATION_IMG_SIZE - mask.height) // 2))

        image_canvas.save(output_image)
        mask_canvas.save(output_mask)


def convert_zenodo_json_dataset(raw_root: Path, output_root: Path = config.SEGMENTATION_DATA_DIR) -> int:
    """Convert Zenodo/Hutton image + same-stem JSON pairs."""
    count = 0
    for annotation_path in raw_root.rglob("*.json"):
        image_path = None
        for candidate in annotation_path.parent.iterdir():
            if candidate.stem == annotation_path.stem and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                image_path = candidate
                break
        if image_path is None:
            continue

        output_image = output_root / "images" / f"{image_path.stem}.jpg"
        output_mask = output_root / "masks" / f"{image_path.stem}.png"
        convert_polygon_pair(image_path, annotation_path, output_image, output_mask)
        count += 1
    print(f"Converted {count} polygon annotations from {raw_root}")
    return count


def load_mask(mask_path: str) -> tf.Tensor:
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE), method="nearest")
    return tf.cast(mask > 127, tf.float32)


def load_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE))
    return tf.cast(image, tf.float32) / 255.0


def list_segmentation_pairs(
    image_dir: Path = config.SEGMENTATION_IMAGE_DIR,
    mask_dir: Path = config.SEGMENTATION_MASK_DIR,
) -> list[tuple[str, str]]:
    """List paired segmentation image/mask files."""
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((str(image_path), str(mask_path)))
    return pairs


def dataset_from_pairs(
    pairs: list[tuple[str, str]],
    batch_size: int = config.SEGMENTATION_BATCH_SIZE,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data segmentation dataset from explicit image/mask pairs."""
    if not pairs:
        raise FileNotFoundError("No image/mask pairs provided")

    image_paths, mask_paths = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))
    if shuffle:
        dataset = dataset.shuffle(len(pairs), seed=config.SEED)
    dataset = dataset.map(lambda x, y: (load_image(x), load_mask(y)), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_segmentation_pairs(
    pairs: list[tuple[str, str]],
    val_fraction: float = 0.2,
    seed: int = config.SEED,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministically split segmentation pairs into train and validation sets."""
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    val_size = max(1, int(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
    return pairs[val_size:], pairs[:val_size]


def create_segmentation_dataset(
    image_dir: Path = config.SEGMENTATION_IMAGE_DIR,
    mask_dir: Path = config.SEGMENTATION_MASK_DIR,
    batch_size: int = config.SEGMENTATION_BATCH_SIZE,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data dataset from paired image/mask files."""
    pairs = list_segmentation_pairs(image_dir, mask_dir)

    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    return dataset_from_pairs(pairs, batch_size=batch_size, shuffle=shuffle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leaf segmentation data.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=config.SEGMENTATION_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_zenodo_json_dataset(args.raw_root, args.output_root)


if __name__ == "__main__":
    main()
