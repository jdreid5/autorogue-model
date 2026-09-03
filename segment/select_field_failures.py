"""Select field canopy images that should be annotated for segmenter recovery."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import config
from field_splits import excluded_canopy_stems


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validation_lookup(validation_results: dict) -> dict[str, dict]:
    lookup = {}
    for split_name in ("adapt", "validation", "test"):
        for record in validation_results.get(split_name, {}).get("predictions", []):
            lookup[record["image_path"]] = record
    return lookup


def score_canopy(canopy: dict, prediction: dict | None) -> tuple[int, list[str]]:
    reasons = []
    score = 0
    crop_count = int(canopy.get("crop_count", 0))
    rejected_count = int(canopy.get("rejected_count", 0))

    if crop_count == 0:
        score += 100
        reasons.append("zero_crop")
    elif crop_count == 1:
        score += 35
        reasons.append("single_crop")
    if rejected_count:
        score += 15
        reasons.append("has_rejections")

    if prediction is not None:
        used_leaf_count = int(prediction.get("used_leaf_count", 0))
        predicted_label = prediction.get("predicted_label")
        true_label = prediction.get("true_label")
        if used_leaf_count == 0:
            score += 75
            reasons.append("zero_confident_crop")
        if predicted_label == "uncertain":
            score += 50
            reasons.append("uncertain")
        elif predicted_label != true_label:
            score += 40
            reasons.append("wrong_prediction")

    return score, reasons


def select_candidates(
    audit: dict,
    validation_results: dict,
    max_count: int,
    held_out_stems: set[str] | None = None,
) -> list[dict]:
    """Rank canopies worth annotating, drawing only from the trainable pool.

    Held-out canopies are excluded outright. Previously they were preferred: a
    canopy scored +10 for sitting in validation or test, which sent annotation
    effort at exactly the canopies the segmenter would later be graded on.
    """
    held_out_stems = held_out_stems or set()
    predictions = validation_lookup(validation_results)
    scored = []
    skipped_held_out = 0
    for canopy in audit.get("canopies", []):
        if Path(canopy["canopy_path"]).stem in held_out_stems:
            skipped_held_out += 1
            continue
        prediction = predictions.get(canopy["canopy_path"])
        score, reasons = score_canopy(canopy, prediction)
        scored.append(
            {
                "canopy_path": canopy["canopy_path"],
                "source_class": canopy["source_class"],
                "unified_class": canopy["unified_class"],
                "split": canopy["split"],
                "crop_count": canopy["crop_count"],
                "rejected_count": canopy["rejected_count"],
                "predicted_label": prediction.get("predicted_label") if prediction else None,
                "used_leaf_count": prediction.get("used_leaf_count") if prediction else None,
                "score": score,
                "reasons": reasons,
            }
        )

    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in sorted(scored, key=lambda item: item["score"], reverse=True):
        by_class[record["unified_class"]].append(record)

    selected = []
    classes = sorted(by_class)
    per_class_target = max(1, max_count // max(1, len(classes)))
    for class_name in classes:
        selected.extend(by_class[class_name][:per_class_target])

    remaining = [
        record
        for record in sorted(scored, key=lambda item: item["score"], reverse=True)
        if record not in selected
    ]
    selected.extend(remaining[: max(0, max_count - len(selected))])
    if skipped_held_out:
        print(f"Skipped {skipped_held_out} held-out canopies; they must not be annotated")
    return selected[:max_count]


def write_contact_sheet(candidates: list[dict], output_path: Path, tile_size: int = 192, columns: int = 5) -> None:
    if not candidates:
        return
    label_height = 38
    rows = (len(candidates) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for index, record in enumerate(candidates):
        image_path = Path(record["canopy_path"])
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            row, col = divmod(index, columns)
            x = col * tile_size + (tile_size - image.width) // 2
            y = row * (tile_size + label_height) + (tile_size - image.height) // 2
            sheet.paste(image, (x, y))
            label_y = row * (tile_size + label_height) + tile_size + 3
            label = f"{record['unified_class']} crops={record['crop_count']} score={record['score']}"
            draw.text((col * tile_size + 4, label_y), label, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select field segmentation failures for annotation.")
    parser.add_argument("--audit-path", type=Path, default=config.FIELD_LEAF_DIR / "crop_audit.json")
    parser.add_argument("--validation-path", type=Path, default=config.OUTPUTS_DIR / "field_validation_results.json")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=config.OUTPUTS_DIR / "field_segmentation_annotation_candidates.json",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=config.OUTPUTS_DIR / "field_segmentation_annotation_candidates.jpg",
    )
    parser.add_argument("--max-count", type=int, default=config.SEGMENTATION_FAILURE_SAMPLE_COUNT)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=config.UNTOUCHED_SPLIT_PATH,
        help="Field split whose held-out canopies must not be offered for annotation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = load_json(args.audit_path)
    validation_results = load_json(args.validation_path)
    candidates = select_candidates(
        audit,
        validation_results,
        args.max_count,
        held_out_stems=excluded_canopy_stems(split_path=args.split_path),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump({"count": len(candidates), "candidates": candidates}, handle, indent=2)
    write_contact_sheet(candidates, args.contact_sheet)

    print(f"Selected {len(candidates)} field canopies for annotation")
    print(f"Candidate manifest written to {args.output_json}")
    print(f"Contact sheet written to {args.contact_sheet}")


if __name__ == "__main__":
    main()
