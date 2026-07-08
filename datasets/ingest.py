"""
Ingest public leaf datasets into Autorogue's unified class structure.

Expected source layout:
  data/raw/<source_name>/<source-specific-class>/*.jpg

The source/class mappings live in config.SOURCE_CLASS_MAP. Images mapped to
"ignore" are skipped. Output files are copied into:
  data/public-leaves/<unified-class>/
with a provenance manifest for source-aware splitting and audit checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class IngestRecord:
    source: str
    original_class: str
    unified_class: str
    original_path: str
    output_path: str


def iter_images(root: Path) -> Iterable[Path]:
    """Yield image files under root recursively."""
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def infer_source_class(image_path: Path, source_root: Path) -> str:
    """Use the first directory below the source root as the source class."""
    relative = image_path.relative_to(source_root)
    if len(relative.parts) < 2:
        return source_root.name
    return relative.parts[0]


def safe_output_name(source: str, source_class: str, image_path: Path) -> str:
    """Create a stable filename that preserves source provenance."""
    safe_class = source_class.replace(" ", "_").replace("/", "_")
    return f"{source}__{safe_class}__{image_path.stem}{image_path.suffix.lower()}"


def safe_crop_output_name(source: str, source_class: str, image_path: Path, object_idx: int) -> str:
    """Create a stable filename for object crops from detection datasets."""
    safe_class = source_class.replace(" ", "_").replace("/", "_")
    return f"{source}__{safe_class}__{image_path.stem}__obj_{object_idx:03d}{image_path.suffix.lower()}"


def copy_normalized_image(src: Path, dst: Path) -> None:
    """Copy an image after applying EXIF orientation and RGB conversion."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(dst)


def crop_normalized_object(src: Path, dst: Path, bbox: tuple[int, int, int, int]) -> None:
    """Crop an object box, apply EXIF orientation, and save RGB."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        left, top, right, bottom = bbox
        left = max(0, min(left, image.width - 1))
        top = max(0, min(top, image.height - 1))
        right = max(left + 1, min(right, image.width))
        bottom = max(top + 1, min(bottom, image.height))
        image.crop((left, top, right, bottom)).save(dst)


def plantdoc_image_for_annotation(annotation_path: Path, source_root: Path) -> Path | None:
    """Find the matching PlantDoc image for a Supervisely-style annotation."""
    split_dir = annotation_path.parent.parent
    image_dir = split_dir / "img"
    image_name = annotation_path.name.removesuffix(".json")
    candidate = image_dir / image_name
    if candidate.exists():
        return candidate
    for ext in IMAGE_EXTENSIONS:
        fallback = image_dir / f"{annotation_path.stem}{ext}"
        if fallback.exists():
            return fallback
    return None


def bbox_from_plantdoc_object(obj: dict) -> tuple[int, int, int, int] | None:
    """Extract a rectangle bbox from PlantDoc/Supervisely object JSON."""
    exterior = obj.get("points", {}).get("exterior", [])
    if len(exterior) < 2:
        return None
    xs = [int(round(point[0])) for point in exterior]
    ys = [int(round(point[1])) for point in exterior]
    return min(xs), min(ys), max(xs), max(ys)


def ingest_plantdoc(source: str, source_root: Path, output_root: Path) -> list[IngestRecord]:
    """Ingest PlantDoc by cropping mapped object annotations from train/test."""
    class_map = config.SOURCE_CLASS_MAP.get(source, {})
    records: list[IngestRecord] = []

    for annotation_path in source_root.glob("**/ann/*.json"):
        image_path = plantdoc_image_for_annotation(annotation_path, source_root)
        if image_path is None:
            print(f"Skipping PlantDoc annotation without image: {annotation_path}")
            continue

        with open(annotation_path, "r", encoding="utf-8") as handle:
            annotation = json.load(handle)

        for object_idx, obj in enumerate(annotation.get("objects", [])):
            source_class = obj.get("classTitle", "")
            unified_class = class_map.get(source_class)
            if unified_class is None:
                continue
            if unified_class == "ignore":
                continue

            bbox = bbox_from_plantdoc_object(obj)
            if bbox is None:
                continue

            output_path = output_root / unified_class / safe_crop_output_name(
                source, source_class, image_path, object_idx
            )
            crop_normalized_object(image_path, output_path, bbox)
            records.append(
                IngestRecord(
                    source=source,
                    original_class=source_class,
                    unified_class=unified_class,
                    original_path=f"{image_path}#{object_idx}",
                    output_path=str(output_path),
                )
            )

    return records


def ingest_source(source: str, source_root: Path, output_root: Path) -> list[IngestRecord]:
    """Ingest one configured dataset source."""
    class_map = config.SOURCE_CLASS_MAP.get(source, {})
    records: list[IngestRecord] = []

    if not source_root.exists():
        print(f"Skipping {source}: {source_root} does not exist")
        return records

    if source == "plantdoc":
        return ingest_plantdoc(source, source_root, output_root)

    for image_path in iter_images(source_root):
        source_class = infer_source_class(image_path, source_root)
        unified_class = class_map.get(source_class)

        if unified_class is None:
            print(f"Skipping unmapped class {source_class!r} from {source}: {image_path}")
            continue
        if unified_class == "ignore":
            continue

        output_path = output_root / unified_class / safe_output_name(source, source_class, image_path)
        copy_normalized_image(image_path, output_path)
        records.append(
            IngestRecord(
                source=source,
                original_class=source_class,
                unified_class=unified_class,
                original_path=str(image_path),
                output_path=str(output_path),
            )
        )

    return records


def write_manifest(records: list[IngestRecord], output_root: Path) -> Path:
    """Write JSONL provenance records used by downstream splitting."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")
    return manifest_path


def ingest_all(output_root: Path = config.PUBLIC_LEAF_DIR) -> list[IngestRecord]:
    """Ingest all configured public leaf datasets."""
    all_records: list[IngestRecord] = []
    for source, source_root in config.DATASET_SOURCES.items():
        all_records.extend(ingest_source(source, source_root, output_root))

    manifest_path = write_manifest(all_records, output_root)
    print(f"Ingested {len(all_records)} images")
    print(f"Manifest written to {manifest_path}")
    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest public potato leaf datasets.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.PUBLIC_LEAF_DIR,
        help="Output directory for unified public leaf images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_all(output_root=args.output_root)


if __name__ == "__main__":
    main()
