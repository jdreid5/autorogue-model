#!/usr/bin/env python3
"""
background_crop.py

Crop the outer N% from all sides of each image in a folder, then center-crop to a square.

Usage:
  python background_crop.py /path/to/input /path/to/output
  # Optional flags:
  #   --border-percent 0.20     (default 0.20 → crop 20% off EACH side; keep center 60%)
  #   --recursive               (process files in subfolders)
  #   --overwrite               (allow overwriting existing files)
  #   --extensions .jpg .jpeg .png .webp
"""

import argparse
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop outer N%% and then center-crop to square.")
    p.add_argument("input_dir", type=Path, help="Folder containing images")
    p.add_argument("output_dir", type=Path, help="Folder to write cropped images")
    p.add_argument("--border-percent", type=float, default=0.20,
                   help="Fraction to crop from EACH side (default: 0.20)")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite files in output folder if they exist")
    p.add_argument("--extensions", nargs="+",
                   default=[".jpg", ".jpeg", ".png", ".webp"],
                   help="File extensions to include (case-insensitive)")
    return p.parse_args()

def list_images(root: Path, recursive: bool, exts: Tuple[str, ...]):
    exts_lower = tuple(e.lower() for e in exts)
    if recursive:
        yield from (p for p in root.rglob("*") if p.suffix.lower() in exts_lower and p.is_file())
    else:
        yield from (p for p in root.iterdir() if p.suffix.lower() in exts_lower and p.is_file())

def crop_outer_percent(img: Image.Image, percent: float) -> Image.Image:
    """
    Crop `percent` from EACH side.
    percent=0.20 means remove left/right/top/bottom 20%, keeping center 60%.
    """
    if percent < 0 or percent >= 0.5:
        raise ValueError("border-percent must be >= 0 and < 0.5")

    w, h = img.size
    dx = int(round(w * percent))
    dy = int(round(h * percent))

    # Ensure we keep at least 1px in each dimension
    left   = min(max(0, dx), w - 1)
    top    = min(max(0, dy), h - 1)
    right  = max(min(w - dx, w), left + 1)
    bottom = max(min(h - dy, h), top + 1)

    return img.crop((left, top, right, bottom))

def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def save_image(img: Image.Image, out_path: Path, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    # Choose reasonable defaults per format
    fmt = (out_path.suffix.lower().lstrip(".") or "png").upper()
    save_kwargs = {}
    if fmt in ("JPG", "JPEG"):
        # preserve progressive JPEG, good quality
        save_kwargs.update(dict(quality=95, optimize=True, progressive=True))
        fmt = "JPEG"
    elif fmt == "PNG":
        save_kwargs.update(dict(optimize=True))
    elif fmt == "WEBP":
        save_kwargs.update(dict(quality=95, method=6))

    img.save(out_path, format=fmt, **save_kwargs)

def process_image(in_path: Path, out_root: Path, base_root: Path,
                  border_percent: float, overwrite: bool):
    # Keep the same relative structure under out_root
    rel = in_path.relative_to(base_root)
    out_path = out_root / rel

    # Ensure EXIF orientation is applied
    with Image.open(in_path) as im0:
        im = ImageOps.exif_transpose(im0.convert("RGB"))

    # Step 1: crop outer N% from each side
    im = crop_outer_percent(im, border_percent)

    # Step 2: center-crop to square
    im = center_square_crop(im)

    save_image(im, out_path, overwrite)
    return out_path

def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir does not exist or is not a directory: {input_dir}")

    exts = tuple(args.extensions)

    count = 0
    skipped = 0
    for img_path in list_images(input_dir, args.recursive, exts):
        try:
            out_path = process_image(
                img_path,
                output_dir,
                base_root=input_dir,
                border_percent=args.border_percent,
                overwrite=args.overwrite,
            )
            print(f"✓ {img_path}  ->  {out_path}")
            count += 1
        except FileExistsError as e:
            print(f"⏭  {img_path} (exists; use --overwrite to replace)")
            skipped += 1
        except Exception as e:
            print(f"✗ {img_path}: {e}")
            skipped += 1

    print(f"\nDone. Processed: {count}, Skipped/Errors: {skipped}")

if __name__ == "__main__":
    main()
