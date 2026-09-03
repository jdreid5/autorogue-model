"""
Create weakly-labeled field leaf crops from plant-labeled canopy images.

Each canopy image inherits its folder label (e.g. healthy-russets) and all leaf
crops segmented from that image receive the mapped unified class label.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

import config
from datasets.ingest import IMAGE_EXTENSIONS
from field_splits import split_lookup
from segment.infer import CropRejection, LeafInstance, load_segmenter, save_leaf_instances, segment_image_with_audit


@dataclass(frozen=True)
class FieldLeafRecord:
    canopy_path: str
    output_path: str
    source_class: str
    unified_class: str
    weak_label: bool
    variety: str
    split: str
    bbox: tuple[int, int, int, int]
    segmentation_bbox: tuple[int, int, int, int]
    area: int
    segmentation_area: int
    crop_width: int
    crop_height: int
    mask_coverage: float
    upsample_factor: float


def infer_variety(source_class: str) -> str:
    parts = source_class.split("-")
    return parts[-1] if len(parts) > 1 else "unknown"


def rejection_to_record(rejection: CropRejection) -> dict:
    return {
        "reason": rejection.reason,
        "bbox": rejection.bbox,
        "segmentation_bbox": rejection.segmentation_bbox,
        "segmentation_area": rejection.segmentation_area,
        "crop_width": rejection.crop_size[0],
        "crop_height": rejection.crop_size[1],
        "mask_coverage": rejection.mask_coverage,
        "upsample_factor": rejection.upsample_factor,
    }


def crop_summary(records: list[FieldLeafRecord]) -> dict:
    if not records:
        return {
            "mean_crop_width": 0.0,
            "mean_crop_height": 0.0,
            "mean_mask_coverage": 0.0,
            "mean_upsample_factor": 0.0,
        }
    return {
        "mean_crop_width": sum(record.crop_width for record in records) / len(records),
        "mean_crop_height": sum(record.crop_height for record in records) / len(records),
        "mean_mask_coverage": sum(record.mask_coverage for record in records) / len(records),
        "mean_upsample_factor": sum(record.upsample_factor for record in records) / len(records),
        "min_crop_width": min(record.crop_width for record in records),
        "min_crop_height": min(record.crop_height for record in records),
        "max_upsample_factor": max(record.upsample_factor for record in records),
    }


def write_contact_sheet(
    records: list[FieldLeafRecord],
    output_root: Path,
    max_samples: int = config.CROP_CONTACT_SHEET_SAMPLES,
) -> Path | None:
    if not records:
        return None

    by_class: dict[str, list[FieldLeafRecord]] = {}
    for record in records:
        by_class.setdefault(record.unified_class, []).append(record)

    samples: list[FieldLeafRecord] = []
    per_class = max(1, max_samples // max(1, len(by_class)))
    for class_name in sorted(by_class):
        samples.extend(by_class[class_name][:per_class])
    samples = samples[:max_samples]

    tile_size = config.IMG_SIZE
    label_height = 26
    columns = 5
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for idx, record in enumerate(samples):
        row, col = divmod(idx, columns)
        x = col * tile_size
        y = row * (tile_size + label_height)
        with Image.open(record.output_path) as crop:
            sheet.paste(crop.convert("RGB"), (x, y))
        label = f"{record.unified_class} {record.crop_width}x{record.crop_height} m={record.mask_coverage:.2f}"
        draw.text((x + 4, y + tile_size + 4), label, fill=(0, 0, 0))

    output_path = output_root / "crop_contact_sheet.jpg"
    sheet.save(output_path, quality=90)
    return output_path


def segment_canopy_dataset(
    canopy_root: Path = config.CANOPY_DIR,
    output_root: Path = config.FIELD_LEAF_DIR,
    model_path: Path | None = None,
    clean_output: bool = False,
    split_path: Path | None = None,
) -> list[FieldLeafRecord]:
    source_map = config.SOURCE_CLASS_MAP["canopy_weak"]
    segmenter = load_segmenter(model_path)
    canopy_splits = split_lookup(split_path=split_path, canopy_root=canopy_root)
    records: list[FieldLeafRecord] = []
    canopy_audit: list[dict] = []
    rejected_components: list[dict] = []

    if clean_output and output_root.exists():
        shutil.rmtree(output_root)

    for source_class, unified_class in source_map.items():
        if unified_class == "ignore":
            continue
        class_dir = canopy_root / source_class
        if not class_dir.exists():
            print(f"Skipping missing canopy class: {class_dir}")
            continue

        output_class_dir = output_root / unified_class
        for image_path in class_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(image_path) as image:
                instances, rejections = segment_image_with_audit(image, model=segmenter)
            saved_paths = save_leaf_instances(instances, output_class_dir, image_path.stem)
            split = canopy_splits.get(str(image_path), "unknown")
            rejection_records = [rejection_to_record(rejection) for rejection in rejections]
            for rejection_record in rejection_records:
                rejected_components.append(
                    {
                        "canopy_path": str(image_path),
                        "source_class": source_class,
                        "unified_class": unified_class,
                        "split": split,
                        **rejection_record,
                    }
                )
            canopy_audit.append(
                {
                    "canopy_path": str(image_path),
                    "source_class": source_class,
                    "unified_class": unified_class,
                    "split": split,
                    "crop_count": len(saved_paths),
                    "rejected_count": len(rejection_records),
                    "rejections": rejection_records,
                }
            )
            for instance, saved_path in zip(instances, saved_paths):
                records.append(
                    FieldLeafRecord(
                        canopy_path=str(image_path),
                        output_path=str(saved_path),
                        source_class=source_class,
                        unified_class=unified_class,
                        weak_label=True,
                        variety=infer_variety(source_class),
                        split=split,
                        bbox=instance.bbox,
                        segmentation_bbox=instance.segmentation_bbox,
                        area=instance.area,
                        segmentation_area=instance.segmentation_area,
                        crop_width=instance.crop_size[0],
                        crop_height=instance.crop_size[1],
                        mask_coverage=instance.mask_coverage,
                        upsample_factor=instance.upsample_factor,
                    )
                )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")

    counts = [record["crop_count"] for record in canopy_audit]
    rejected_counts = [record["rejected_count"] for record in canopy_audit]
    audit = {
        "canopy_count": len(canopy_audit),
        "crop_count": len(records),
        "rejected_component_count": sum(rejected_counts),
        "canopies_with_crops": sum(1 for count in counts if count > 0),
        "zero_crop_canopies": sum(1 for count in counts if count == 0),
        "mean_crops_per_canopy": (sum(counts) / len(counts)) if counts else 0.0,
        "mean_rejected_components_per_canopy": (sum(rejected_counts) / len(rejected_counts)) if rejected_counts else 0.0,
        "crop_count_histogram": dict(Counter(counts)),
        "rejection_reasons": dict(Counter(record["reason"] for record in rejected_components)),
        "by_split": dict(Counter(record["split"] for record in canopy_audit)),
        "by_class": dict(Counter(record.unified_class for record in records)),
        "crop_quality": crop_summary(records),
        "canopies": canopy_audit,
    }
    audit_path = output_root / "crop_audit.json"
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    contact_sheet_path = write_contact_sheet(records, output_root)

    print(f"Created {len(records)} weakly-labeled field leaf crops")
    print(f"Manifest written to {manifest_path}")
    print(f"Crop audit written to {audit_path}")
    if contact_sheet_path:
        print(f"Crop contact sheet written to {contact_sheet_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment canopy images into weakly-labeled leaves.")
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--output-root", type=Path, default=config.FIELD_LEAF_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--clean-output", action="store_true", help="Remove old field crops before regenerating.")
    parser.add_argument("--split-path", type=Path, default=config.UNTOUCHED_SPLIT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segment_canopy_dataset(
        args.canopy_root,
        args.output_root,
        args.model_path,
        args.clean_output,
        split_path=args.split_path,
    )


if __name__ == "__main__":
    main()
