"""
Harmonize public leaf crops before classifier training.

This script standardizes size and neutralizes obvious backgrounds so public
plain/black-background leaves and segmented field leaves are closer in
appearance. It intentionally uses a conservative color/brightness heuristic;
later runs can replace the mask with outputs from the trained segmenter.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import config
from datasets.ingest import IMAGE_EXTENSIONS

NEUTRAL_BACKGROUND = (128, 128, 128)


def estimate_leaf_mask(image: Image.Image) -> Image.Image:
    """Estimate a coarse leaf mask using green dominance and saturation."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_channel = rgb.max(axis=-1)
    min_channel = rgb.min(axis=-1)
    saturation = max_channel - min_channel

    green_dominant = (green > red * 0.9) & (green > blue * 0.9)
    sufficiently_colored = saturation > 18
    not_too_dark = max_channel > 35
    mask = green_dominant & sufficiently_colored & not_too_dark

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    return mask_img.filter(ImageFilter.GaussianBlur(config.MASK_FEATHER_RADIUS))


def normalize_background(image: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """Composite the image over a neutral background using the leaf mask."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mask is None:
        mask = estimate_leaf_mask(image)

    background = Image.new("RGB", image.size, NEUTRAL_BACKGROUND)
    return Image.composite(image, background, mask)


def fit_square(image: Image.Image, size: int) -> Image.Image:
    """Resize preserving aspect ratio and pad to a square canvas."""
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), NEUTRAL_BACKGROUND)
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def harmonize_image(src: Path, dst: Path, image_size: int = config.IMG_SIZE) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = normalize_background(image)
        image = fit_square(image, image_size)
        image.save(dst)


def copy_manifest(input_root: Path, output_root: Path) -> None:
    src_manifest = input_root / "manifest.jsonl"
    if src_manifest.exists():
        shutil.copy2(src_manifest, output_root / "manifest.jsonl")


def write_harmonize_info(output_root: Path) -> None:
    info = {
        "image_size": config.IMG_SIZE,
        "background": NEUTRAL_BACKGROUND,
        "method": "green-dominance coarse mask with neutral background composite",
        "classes": config.CLASSES,
    }
    with open(output_root / "harmonize_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def harmonize_dataset(
    input_root: Path = config.PUBLIC_LEAF_DIR,
    output_root: Path = config.HARMONIZED_LEAF_DIR,
    image_size: int = config.IMG_SIZE,
) -> int:
    """Harmonize every image in class subdirectories."""
    count = 0
    for class_name in config.CLASSES:
        class_dir = input_root / class_name
        if not class_dir.exists():
            continue
        for src in class_dir.rglob("*"):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = src.relative_to(input_root)
            dst = output_root / rel
            harmonize_image(src, dst, image_size=image_size)
            count += 1

    output_root.mkdir(parents=True, exist_ok=True)
    copy_manifest(input_root, output_root)
    write_harmonize_info(output_root)
    print(f"Harmonized {count} images into {output_root}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harmonize public leaf datasets.")
    parser.add_argument("--input-root", type=Path, default=config.PUBLIC_LEAF_DIR)
    parser.add_argument("--output-root", type=Path, default=config.HARMONIZED_LEAF_DIR)
    parser.add_argument("--image-size", type=int, default=config.IMG_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    harmonize_dataset(args.input_root, args.output_root, args.image_size)


if __name__ == "__main__":
    main()
