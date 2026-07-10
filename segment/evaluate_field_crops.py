"""Evaluate field canopy crop quality before classifier adaptation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import config
from datasets.ingest import IMAGE_EXTENSIONS
from field_splits import split_lookup
from segment.infer import CropRejection, LeafInstance, load_segmenter, segment_image_with_audit


def crop_record(instance: LeafInstance) -> dict:
    return {
        "bbox": instance.bbox,
        "area": instance.area,
        "segmentation_bbox": instance.segmentation_bbox,
        "segmentation_area": instance.segmentation_area,
        "crop_size": instance.crop_size,
        "mask_coverage": instance.mask_coverage,
        "upsample_factor": instance.upsample_factor,
    }


def iter_canopy_images(canopy_root: Path) -> list[tuple[Path, str, str]]:
    records = []
    source_map = config.SOURCE_CLASS_MAP["canopy_weak"]
    for source_class, unified_class in source_map.items():
        if unified_class == "ignore":
            continue
        class_dir = canopy_root / source_class
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((image_path, source_class, unified_class))
    return records


def instance_summary(instances: list[LeafInstance]) -> dict:
    if not instances:
        return {
            "mean_crop_width": 0.0,
            "mean_crop_height": 0.0,
            "mean_mask_coverage": 0.0,
            "mean_upsample_factor": 0.0,
        }
    return {
        "mean_crop_width": sum(instance.crop_size[0] for instance in instances) / len(instances),
        "mean_crop_height": sum(instance.crop_size[1] for instance in instances) / len(instances),
        "mean_mask_coverage": sum(instance.mask_coverage for instance in instances) / len(instances),
        "mean_upsample_factor": sum(instance.upsample_factor for instance in instances) / len(instances),
        "min_crop_width": min(instance.crop_size[0] for instance in instances),
        "min_crop_height": min(instance.crop_size[1] for instance in instances),
        "max_upsample_factor": max(instance.upsample_factor for instance in instances),
    }


def draw_overlay(image_path: Path, instances: list[LeafInstance], rejections: list[CropRejection], tile_size: int) -> Image.Image:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    scale = min(tile_size / original_width, tile_size / original_height)
    resized = image.resize((max(1, int(original_width * scale)), max(1, int(original_height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
    offset = ((tile_size - resized.width) // 2, (tile_size - resized.height) // 2)
    canvas.paste(resized, offset)
    draw = ImageDraw.Draw(canvas)

    def project_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (
            int(box[0] * scale) + offset[0],
            int(box[1] * scale) + offset[1],
            int(box[2] * scale) + offset[0],
            int(box[3] * scale) + offset[1],
        )

    for instance in instances:
        draw.rectangle(project_box(instance.bbox), outline=(0, 180, 0), width=3)
    for rejection in rejections:
        if rejection.bbox is not None:
            draw.rectangle(project_box(rejection.bbox), outline=(210, 80, 0), width=2)
    return canvas


def write_overlay_sheet(records: list[dict], output_path: Path, max_samples: int, tile_size: int = 192) -> None:
    if not records:
        return
    samples = sorted(records, key=lambda record: (record["crop_count"], -record["rejected_count"]))[:max_samples]
    columns = 5
    label_height = 42
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for idx, record in enumerate(samples):
        tile = draw_overlay(
            Path(record["canopy_path"]),
            record.pop("_instances"),
            record.pop("_rejections"),
            tile_size,
        )
        row, col = divmod(idx, columns)
        x = col * tile_size
        y = row * (tile_size + label_height)
        sheet.paste(tile, (x, y))
        label = f"{record['unified_class']} crops={record['crop_count']} rej={record['rejected_count']}"
        draw.text((x + 4, y + tile_size + 4), label, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def evaluate_field_crops(
    canopy_root: Path = config.CANOPY_DIR,
    model_path: Path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME,
    output_path: Path = config.OUTPUTS_DIR / "field_crop_quality.json",
    contact_sheet_path: Path = config.OUTPUTS_DIR / "field_crop_quality_overlays.jpg",
    max_sheet_samples: int = config.CROP_CONTACT_SHEET_SAMPLES,
) -> dict:
    segmenter = load_segmenter(model_path)
    canopy_splits = split_lookup(canopy_root=canopy_root)
    records = []
    all_instances: list[LeafInstance] = []
    rejection_reasons: Counter[str] = Counter()

    for image_path, source_class, unified_class in iter_canopy_images(canopy_root):
        with Image.open(image_path) as image:
            instances, rejections = segment_image_with_audit(image, model=segmenter)
        all_instances.extend(instances)
        rejection_reasons.update(rejection.reason for rejection in rejections)
        records.append(
            {
                "canopy_path": str(image_path),
                "source_class": source_class,
                "unified_class": unified_class,
                "split": canopy_splits.get(str(image_path), "unknown"),
                "crop_count": len(instances),
                "rejected_count": len(rejections),
                "rejections": [asdict(rejection) for rejection in rejections],
                "crops": [crop_record(instance) for instance in instances],
                "_instances": instances,
                "_rejections": rejections,
            }
        )

    crop_counts = [record["crop_count"] for record in records]
    rejected_counts = [record["rejected_count"] for record in records]
    serializable_records = [{key: value for key, value in record.items() if not key.startswith("_")} for record in records]
    results = {
        "canopy_count": len(records),
        "crop_count": sum(crop_counts),
        "rejected_component_count": sum(rejected_counts),
        "canopies_with_crops": sum(1 for count in crop_counts if count > 0),
        "zero_crop_canopies": sum(1 for count in crop_counts if count == 0),
        "zero_crop_rate": (sum(1 for count in crop_counts if count == 0) / len(crop_counts)) if crop_counts else 0.0,
        "mean_crops_per_canopy": (sum(crop_counts) / len(crop_counts)) if crop_counts else 0.0,
        "crop_count_histogram": dict(Counter(crop_counts)),
        "rejection_reasons": dict(rejection_reasons),
        "by_split": dict(Counter(record["split"] for record in records)),
        "by_class": dict(Counter(record["unified_class"] for record in records for _ in range(record["crop_count"]))),
        "crop_quality": instance_summary(all_instances),
        "canopies": serializable_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_overlay_sheet(records, contact_sheet_path, max_sheet_samples)
    print(f"Field crop quality written to {output_path}")
    print(f"Overlay contact sheet written to {contact_sheet_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmenter crop quality on field canopies.")
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--output-path", type=Path, default=config.OUTPUTS_DIR / "field_crop_quality.json")
    parser.add_argument("--contact-sheet-path", type=Path, default=config.OUTPUTS_DIR / "field_crop_quality_overlays.jpg")
    parser.add_argument("--max-sheet-samples", type=int, default=config.CROP_CONTACT_SHEET_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_field_crops(
        canopy_root=args.canopy_root,
        model_path=args.model_path,
        output_path=args.output_path,
        contact_sheet_path=args.contact_sheet_path,
        max_sheet_samples=args.max_sheet_samples,
    )


if __name__ == "__main__":
    main()
