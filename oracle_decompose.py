"""
Oracle error decomposition for the segment-then-classify field pipeline.

Runs three arms over the human-annotated field canopies and attributes the
end-to-end failure to segmentation, classification, or lost canopy context:

    A. oracle    human leaf polygon -> crop -> classifier
    B. segmenter segmenter component -> crop -> classifier
    C. canopy    whole canopy image -> classifier

Arm A minus arm B isolates segmentation loss, arm A on its own bounds what leaf
level classification can do with perfect crops, and arm C minus arm A shows
whether cropping to single leaves discards diagnostic canopy context.

Also audits segmenter crops against the human polygons for completeness,
purity, multi-leaf merges, and instance fragmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config
from field_splits import load_field_splits
from field_validate import aggregate_with_thresholds
from pipeline.infer_canopy import LeafPrediction, classify_instances, load_classifier
from segment.data import (
    list_combined_segmentation_pairs,
    load_generic_polygon_json,
    rasterize_polygons,
    split_segmentation_pairs,
)
from segment.infer import (
    LeafInstance,
    crop_reject_reason,
    load_segmenter,
    segment_image_with_audit,
)
from segment.postprocess import feather_mask

ARMS = ("oracle", "segmenter", "canopy")
FIELD_CLASSES = ("healthy", "leaf_roll")

# Audit geometry is evaluated on a downscaled canvas; the segmenter itself only
# resolves 384px, so full 2150px rasterization would add cost without accuracy.
AUDIT_CANVAS = 512
MATCH_IOU = 0.5
LOOSE_MATCH_IOU = 0.25
# A crop "contains" a human leaf when it covers at least this much of its area.
MERGE_COVERAGE = 0.30
# A human leaf is "fragmented" when several crops each take this much of it.
SPLIT_CONTRIBUTION = 0.25

THRESHOLD_GRID = [round(float(value), 2) for value in np.arange(0.05, 0.55, 0.05)]


@dataclass
class CanopyRecord:
    canopy_path: str
    annotation_path: str
    stem: str
    true_label: str
    source_class: str
    field_split: str
    segmenter_split: str
    human_instance_count: int
    arm_predictions: dict[str, dict[str, list[LeafPrediction]]] = dataclass_field(default_factory=dict)
    crop_counts: dict[str, int] = dataclass_field(default_factory=dict)
    segmenter_rejections: list[str] = dataclass_field(default_factory=list)
    oracle_gate_rejections: list[str] = dataclass_field(default_factory=list)
    audit: dict = dataclass_field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------


def canopy_path_for_annotation(annotation_path: Path, canopy_root: Path) -> Path | None:
    """Invert `segment.prepare_field_annotation_set.safe_annotation_name`."""
    name = annotation_path.stem
    if "__" not in name:
        return None
    source_token, stem = name.split("__", 1)
    source_class = source_token.replace("_", "-")
    for suffix in (".jpg", ".jpeg", ".png", ".JPG"):
        candidate = canopy_root / source_class / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def reconstruct_segmenter_splits() -> dict[str, str]:
    """Rebuild the deterministic segmenter train/val split by annotation stem.

    Describes the checkpoint that was actually trained, so it deliberately applies
    no held-out exclusions.
    """
    pairs, _ = list_combined_segmentation_pairs(excluded_canopy_stems=None)
    train_pairs, val_pairs = split_segmentation_pairs(pairs, val_fraction=0.2)
    membership: dict[str, str] = {}
    for pairs_subset, name in ((train_pairs, "train"), (val_pairs, "validation")):
        for pair in pairs_subset:
            source = getattr(pair, "source", "source")
            if source != "field":
                continue
            membership[Path(pair.image_path).stem] = name
    return membership


def build_cohort(annotation_dir: Path, canopy_root: Path, split_path: Path | None) -> list[CanopyRecord]:
    field_splits = {}
    for example in load_field_splits(split_path=split_path, canopy_root=canopy_root):
        field_splits[Path(example.image_path).name] = example

    segmenter_splits = reconstruct_segmenter_splits()

    records: list[CanopyRecord] = []
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        if annotation_path.name == "annotation_manifest.json":
            continue
        canopy_path = canopy_path_for_annotation(annotation_path, canopy_root)
        if canopy_path is None:
            print(f"  skipping {annotation_path.name}: no matching canopy image")
            continue
        example = field_splits.get(canopy_path.name)
        if example is None:
            print(f"  skipping {annotation_path.name}: canopy not in field split file")
            continue
        polygons = load_generic_polygon_json(annotation_path)
        records.append(
            CanopyRecord(
                canopy_path=str(canopy_path),
                annotation_path=str(annotation_path),
                stem=annotation_path.stem,
                true_label=example.true_label,
                source_class=example.source_class,
                field_split=example.split,
                segmenter_split=segmenter_splits.get(annotation_path.stem, "unknown"),
                human_instance_count=len([p for p in polygons if len(p) >= 3]),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Crop construction
# ---------------------------------------------------------------------------


def oracle_instance_from_polygon(
    original_image: Image.Image,
    polygon: list[list[float]],
) -> tuple[LeafInstance, str | None] | None:
    """Build a classifier-ready crop from one human polygon, mirroring crop_component."""
    mask = rasterize_polygons(original_image.size, [polygon])
    bbox = mask.getbbox()
    if bbox is None:
        return None

    mask_crop = mask.crop(bbox)
    crop_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    mask_coverage = float(np.asarray(mask_crop).mean() / 255.0)
    upsample_factor = config.IMG_SIZE / max(crop_size) if max(crop_size) > 0 else 0.0
    reject_reason = crop_reject_reason(crop_size, mask_coverage, upsample_factor)

    feathered = feather_mask(np.asarray(mask_crop) > 127)
    crop = original_image.crop(bbox)
    background = Image.new("RGB", crop.size, config.NEUTRAL_BACKGROUND_RGB)
    crop = Image.composite(crop, background, feathered)

    instance = LeafInstance(
        crop=crop,
        mask=feathered,
        bbox=bbox,
        area=int(np.asarray(feathered).sum() / 255),
        segmentation_bbox=bbox,
        segmentation_area=int((np.asarray(mask_crop) > 127).sum()),
        crop_size=crop_size,
        mask_coverage=mask_coverage,
        upsample_factor=float(upsample_factor),
    )
    return instance, reject_reason


def whole_canopy_instance(original_image: Image.Image) -> LeafInstance:
    width, height = original_image.size
    solid = Image.new("L", original_image.size, 255)
    return LeafInstance(
        crop=original_image.copy(),
        mask=solid,
        bbox=(0, 0, width, height),
        area=width * height,
        segmentation_bbox=(0, 0, width, height),
        segmentation_area=width * height,
        crop_size=(width, height),
        mask_coverage=1.0,
        upsample_factor=config.IMG_SIZE / max(width, height),
    )


# ---------------------------------------------------------------------------
# Segmentation audit
# ---------------------------------------------------------------------------


def audit_canvas_geometry(size: tuple[int, int]) -> tuple[tuple[int, int], float]:
    width, height = size
    scale = AUDIT_CANVAS / max(width, height)
    return (max(1, round(width * scale)), max(1, round(height * scale))), scale


def human_masks_on_canvas(
    polygons: list[list[list[float]]],
    canvas_size: tuple[int, int],
    scale: float,
) -> list[np.ndarray]:
    masks = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        scaled = [[float(x) * scale, float(y) * scale] for x, y in polygon]
        mask = np.asarray(rasterize_polygons(canvas_size, [scaled])) > 127
        if mask.any():
            masks.append(mask)
    return masks


def instance_mask_on_canvas(
    instance: LeafInstance,
    canvas_size: tuple[int, int],
    scale: float,
) -> np.ndarray:
    canvas = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
    left, top, right, bottom = instance.bbox
    scaled_left = min(canvas_size[0] - 1, max(0, int(round(left * scale))))
    scaled_top = min(canvas_size[1] - 1, max(0, int(round(top * scale))))
    scaled_right = min(canvas_size[0], max(scaled_left + 1, int(round(right * scale))))
    scaled_bottom = min(canvas_size[1], max(scaled_top + 1, int(round(bottom * scale))))

    resized = instance.mask.resize(
        (scaled_right - scaled_left, scaled_bottom - scaled_top),
        Image.Resampling.NEAREST,
    )
    canvas[scaled_top:scaled_bottom, scaled_left:scaled_right] = np.asarray(resized) > 127
    return canvas


def audit_segmentation(
    human_masks: list[np.ndarray],
    crop_masks: list[np.ndarray],
) -> dict:
    """Compare accepted segmenter crops against human leaf instances."""
    n_human = len(human_masks)
    n_crops = len(crop_masks)
    human_areas = [int(mask.sum()) for mask in human_masks]
    crop_areas = [int(mask.sum()) for mask in crop_masks]

    iou = np.zeros((n_crops, n_human), dtype=np.float32)
    overlap = np.zeros((n_crops, n_human), dtype=np.int64)
    for i, crop_mask in enumerate(crop_masks):
        for j, human_mask in enumerate(human_masks):
            intersection = int(np.logical_and(crop_mask, human_mask).sum())
            overlap[i, j] = intersection
            if intersection:
                union = crop_areas[i] + human_areas[j] - intersection
                iou[i, j] = intersection / union if union else 0.0

    def greedy_matches(threshold: float) -> list[tuple[int, int]]:
        pairs = [
            (float(iou[i, j]), i, j)
            for i in range(n_crops)
            for j in range(n_human)
            if iou[i, j] >= threshold
        ]
        pairs.sort(reverse=True)
        used_crops: set[int] = set()
        used_human: set[int] = set()
        matches = []
        for _, i, j in pairs:
            if i in used_crops or j in used_human:
                continue
            used_crops.add(i)
            used_human.add(j)
            matches.append((i, j))
        return matches

    strict = greedy_matches(MATCH_IOU)
    loose = greedy_matches(LOOSE_MATCH_IOU)

    # Per-human-leaf completeness.
    instance_coverage = []
    union_coverage = []
    fragmented = 0
    detected = 0
    crop_union = np.logical_or.reduce(crop_masks) if crop_masks else None
    for j, human_mask in enumerate(human_masks):
        area = human_areas[j] or 1
        coverages = [overlap[i, j] / area for i in range(n_crops)]
        best = max(coverages) if coverages else 0.0
        instance_coverage.append(float(best))
        if best >= 0.5:
            detected += 1
        if sum(1 for value in coverages if value >= SPLIT_CONTRIBUTION) >= 2:
            fragmented += 1
        if crop_union is not None:
            union_coverage.append(float(np.logical_and(crop_union, human_mask).sum() / area))
        else:
            union_coverage.append(0.0)

    # Per-crop purity and merge behaviour.
    purity = []
    leaf_pixel_purity = []
    background_fraction = []
    merged_crops = 0
    leaves_per_crop = []
    spurious_crops = 0
    for i in range(n_crops):
        area = crop_areas[i] or 1
        overlaps = overlap[i]
        total_leaf = int(overlaps.sum())
        dominant = int(overlaps.max()) if n_human else 0
        purity.append(float(dominant / area))
        leaf_pixel_purity.append(float(dominant / total_leaf) if total_leaf else 0.0)
        background_fraction.append(float(1.0 - total_leaf / area))
        covered = sum(
            1
            for j in range(n_human)
            if human_areas[j] and overlaps[j] / human_areas[j] >= MERGE_COVERAGE
        )
        leaves_per_crop.append(covered)
        if covered >= 2:
            merged_crops += 1
        if total_leaf == 0 or (n_human and iou[i].max() < LOOSE_MATCH_IOU and dominant / area < 0.5):
            spurious_crops += 1

    human_union = np.logical_or.reduce(human_masks) if human_masks else None
    if human_union is not None and crop_union is not None:
        intersection = float(np.logical_and(human_union, crop_union).sum())
        denominator = float(human_union.sum() + crop_union.sum())
        semantic_dice = (2 * intersection / denominator) if denominator else 0.0
    else:
        intersection = 0.0
        semantic_dice = 0.0

    # Human polygons annotate a subset of the leaves in each canopy, so the
    # union area fractions are needed to tell "crop covers background" apart
    # from "crop covers a real but unannotated leaf".
    canvas_area = float(human_masks[0].size) if human_masks else (float(crop_masks[0].size) if crop_masks else 1.0)
    human_union_area = float(human_union.sum()) if human_union is not None else 0.0
    crop_union_area = float(crop_union.sum()) if crop_union is not None else 0.0

    return {
        "human_union_area_fraction": human_union_area / canvas_area,
        "crop_union_area_fraction": crop_union_area / canvas_area,
        "crop_union_on_annotated_leaf": (intersection / crop_union_area) if crop_union_area else 0.0,
        "annotated_leaf_covered_by_crops": (intersection / human_union_area) if human_union_area else 0.0,
        "human_instances": n_human,
        "accepted_crops": n_crops,
        "matched_iou50": len(strict),
        "matched_iou25": len(loose),
        "detected_50pct_coverage": detected,
        "fragmented_instances": fragmented,
        "merged_crops": merged_crops,
        "spurious_crops": spurious_crops,
        "mean_leaves_per_crop": float(np.mean(leaves_per_crop)) if leaves_per_crop else 0.0,
        "mean_best_instance_coverage": float(np.mean(instance_coverage)) if instance_coverage else 0.0,
        "mean_union_instance_coverage": float(np.mean(union_coverage)) if union_coverage else 0.0,
        "mean_crop_purity": float(np.mean(purity)) if purity else 0.0,
        "mean_leaf_pixel_purity": float(np.mean(leaf_pixel_purity)) if leaf_pixel_purity else 0.0,
        "mean_background_fraction": float(np.mean(background_fraction)) if background_fraction else 0.0,
        "mean_matched_iou": float(np.mean([iou[i, j] for i, j in strict])) if strict else 0.0,
        "semantic_dice": float(semantic_dice),
    }


def render_overlay(
    original: Image.Image,
    polygons: list[list[list[float]]],
    instances: list[LeafInstance],
    tile_size: int = 320,
) -> Image.Image:
    """Green outlines are human leaves, red boxes are accepted segmenter crops."""
    scale = tile_size / max(original.size)
    tile = original.resize(
        (max(1, round(original.width * scale)), max(1, round(original.height * scale))),
        Image.Resampling.LANCZOS,
    ).convert("RGB")

    tint = Image.new("RGB", tile.size, (255, 64, 64))
    crop_union = Image.new("L", tile.size, 0)
    for instance in instances:
        left, top, right, bottom = instance.bbox
        box = (
            max(0, round(left * scale)),
            max(0, round(top * scale)),
            min(tile.width, max(round(left * scale) + 1, round(right * scale))),
            min(tile.height, max(round(top * scale) + 1, round(bottom * scale))),
        )
        resized = instance.mask.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.NEAREST)
        crop_union.paste(resized, box, resized)
    tile = Image.blend(tile, Image.composite(tint, tile, crop_union), 0.45)

    draw = ImageDraw.Draw(tile)
    for instance in instances:
        left, top, right, bottom = instance.bbox
        draw.rectangle(
            [round(left * scale), round(top * scale), round(right * scale), round(bottom * scale)],
            outline=(255, 40, 40),
            width=1,
        )
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = [(float(x) * scale, float(y) * scale) for x, y in polygon]
        draw.line(points + [points[0]], fill=(40, 255, 90), width=2)
    return tile


def write_overlay_sheet(tiles: list[tuple[str, Image.Image]], output_path: Path, columns: int = 4) -> None:
    if not tiles:
        return
    tile_width = max(tile.width for _, tile in tiles)
    tile_height = max(tile.height for _, tile in tiles)
    label_height = 16
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, tile) in enumerate(tiles):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (tile_height + label_height)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + tile_height + 2), label[:64], fill=(220, 220, 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    print(f"Overlay contact sheet saved to {output_path}")


def aggregate_audit(records: list[CanopyRecord]) -> dict:
    audits = [record.audit for record in records if record.audit]
    if not audits:
        return {}

    total_human = sum(audit["human_instances"] for audit in audits)
    total_crops = sum(audit["accepted_crops"] for audit in audits)
    total_matched = sum(audit["matched_iou50"] for audit in audits)
    total_matched_loose = sum(audit["matched_iou25"] for audit in audits)

    def mean_of(key: str) -> float:
        values = [audit[key] for audit in audits]
        return float(np.mean(values)) if values else 0.0

    return {
        "canopies": len(audits),
        "human_instances": total_human,
        "accepted_crops": total_crops,
        "instance_recall_iou50": float(total_matched / total_human) if total_human else 0.0,
        "instance_recall_iou25": float(total_matched_loose / total_human) if total_human else 0.0,
        "instance_precision_iou50": float(total_matched / total_crops) if total_crops else 0.0,
        "detection_rate_50pct_coverage": float(
            sum(audit["detected_50pct_coverage"] for audit in audits) / total_human
        )
        if total_human
        else 0.0,
        "fragmented_instance_rate": float(
            sum(audit["fragmented_instances"] for audit in audits) / total_human
        )
        if total_human
        else 0.0,
        "merged_crop_rate": float(sum(audit["merged_crops"] for audit in audits) / total_crops)
        if total_crops
        else 0.0,
        "spurious_crop_rate": float(sum(audit["spurious_crops"] for audit in audits) / total_crops)
        if total_crops
        else 0.0,
        "mean_leaves_per_crop": mean_of("mean_leaves_per_crop"),
        "mean_best_instance_coverage": mean_of("mean_best_instance_coverage"),
        "mean_union_instance_coverage": mean_of("mean_union_instance_coverage"),
        "mean_crop_purity": mean_of("mean_crop_purity"),
        "mean_leaf_pixel_purity": mean_of("mean_leaf_pixel_purity"),
        "mean_background_fraction": mean_of("mean_background_fraction"),
        "mean_matched_iou": mean_of("mean_matched_iou"),
        "mean_semantic_dice": mean_of("semantic_dice"),
        "crops_per_human_instance": float(total_crops / total_human) if total_human else 0.0,
        "mean_human_union_area_fraction": mean_of("human_union_area_fraction"),
        "mean_crop_union_area_fraction": mean_of("crop_union_area_fraction"),
        "mean_crop_union_on_annotated_leaf": mean_of("crop_union_on_annotated_leaf"),
        "mean_annotated_leaf_covered_by_crops": mean_of("annotated_leaf_covered_by_crops"),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def binary_leaf_roll_probability(prediction: LeafPrediction) -> float:
    healthy = prediction.probabilities.get("healthy", 0.0)
    leaf_roll = prediction.probabilities.get("leaf_roll", 0.0)
    total = healthy + leaf_roll
    return float(leaf_roll / total) if total > 0 else 0.5


def safe_auc(y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def best_operating_point(y_true: list[int], scores: list[float]) -> dict:
    """Best accuracy reachable by moving the decision threshold on a fixed score.

    Separates ranking quality from decision-rule calibration: the gap between
    this and the accuracy at 0.5 is what re-thresholding alone could recover.
    """
    if not scores or len(set(y_true)) < 2:
        return {}
    candidates = sorted(set(scores))
    midpoints = [candidates[0] - 1e-6] + [
        (candidates[i] + candidates[i + 1]) / 2 for i in range(len(candidates) - 1)
    ] + [candidates[-1] + 1e-6]

    best = {"accuracy": -1.0}
    positives = sum(y_true)
    negatives = len(y_true) - positives
    for threshold in midpoints:
        predictions = [1 if score >= threshold else 0 for score in scores]
        correct = sum(1 for p, t in zip(predictions, y_true) if p == t)
        accuracy = correct / len(y_true)
        if accuracy > best["accuracy"]:
            sensitivity = sum(1 for p, t in zip(predictions, y_true) if t == 1 and p == 1) / max(1, positives)
            specificity = sum(1 for p, t in zip(predictions, y_true) if t == 0 and p == 0) / max(1, negatives)
            best = {
                "accuracy": float(accuracy),
                "threshold": float(threshold),
                "leaf_roll_recall": float(sensitivity),
                "healthy_recall": float(specificity),
                "balanced_accuracy": float((sensitivity + specificity) / 2),
            }
    return best


def arm_metrics(
    records: list[CanopyRecord],
    classifier_key: str,
    arm: str,
    thresholds_by_name: dict[str, dict[str, float]],
) -> dict:
    y_true = [record.true_label for record in records]
    predictions_by_canopy = [record.arm_predictions[classifier_key][arm] for record in records]

    # Three-class hard-vote verdicts under each threshold setting.
    hard_vote = {}
    for name, thresholds in thresholds_by_name.items():
        y_pred = [aggregate_with_thresholds(preds, thresholds) for preds in predictions_by_canopy]
        hard_vote[name] = {
            "thresholds": thresholds,
            "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=list(config.CLASSES), average="macro", zero_division=0)
            ),
            "leaf_roll_recall": float(
                sum(1 for t, p in zip(y_true, y_pred) if t == "leaf_roll" and p == "leaf_roll")
                / max(1, sum(1 for t in y_true if t == "leaf_roll"))
            ),
            "prediction_counts": dict(Counter(y_pred)),
        }

    # Best achievable thresholds on this cohort: an upper bound on what
    # aggregation calibration alone could buy.
    best = {"macro_f1": -1.0}
    for leaf_roll_threshold in THRESHOLD_GRID:
        for mosaic_threshold in THRESHOLD_GRID:
            trial = {"leaf_roll": leaf_roll_threshold, "mosaic": mosaic_threshold}
            y_pred = [aggregate_with_thresholds(preds, trial) for preds in predictions_by_canopy]
            score = float(
                f1_score(y_true, y_pred, labels=list(config.CLASSES), average="macro", zero_division=0)
            )
            if score > best["macro_f1"]:
                best = {
                    "macro_f1": score,
                    "thresholds": trial,
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                }
    hard_vote["oracle_best_on_cohort"] = best

    # Binary healthy vs leaf_roll: mosaic has no field examples, so restricting
    # to the two classes that exist removes an artefact of the label space.
    canopy_scores = []
    for preds in predictions_by_canopy:
        values = [binary_leaf_roll_probability(pred) for pred in preds]
        canopy_scores.append(float(np.mean(values)) if values else 0.5)
    binary_truth = [1 if label == "leaf_roll" else 0 for label in y_true]
    soft_pred = ["leaf_roll" if score >= 0.5 else "healthy" for score in canopy_scores]

    vote_fractions = []
    for preds in predictions_by_canopy:
        votes = [1 if binary_leaf_roll_probability(pred) >= 0.5 else 0 for pred in preds]
        vote_fractions.append(float(np.mean(votes)) if votes else 0.5)
    vote_pred = ["leaf_roll" if fraction >= 0.5 else "healthy" for fraction in vote_fractions]

    # Crop-level behaviour under the weak canopy label.
    crop_scores = []
    crop_truth = []
    crop_correct_3class = 0
    crop_total = 0
    confidences = []
    used_flags = []
    for record, preds in zip(records, predictions_by_canopy):
        for pred in preds:
            crop_scores.append(binary_leaf_roll_probability(pred))
            crop_truth.append(1 if record.true_label == "leaf_roll" else 0)
            crop_correct_3class += int(pred.predicted_class == record.true_label)
            crop_total += 1
            confidences.append(pred.confidence)
            used_flags.append(pred.used_for_aggregation)

    return {
        "n_canopies": len(records),
        "n_crops": crop_total,
        "mean_crops_per_canopy": float(crop_total / len(records)) if records else 0.0,
        "canopies_with_zero_crops": int(sum(1 for preds in predictions_by_canopy if not preds)),
        "canopies_with_zero_confident_crops": int(
            sum(1 for preds in predictions_by_canopy if not any(p.used_for_aggregation for p in preds))
        ),
        "hard_vote_3class": hard_vote,
        "binary_soft_vote": {
            "accuracy": float(accuracy_score(y_true, soft_pred)),
            "macro_f1": float(
                f1_score(y_true, soft_pred, labels=list(FIELD_CLASSES), average="macro", zero_division=0)
            ),
            "leaf_roll_recall": float(
                sum(1 for t, p in zip(y_true, soft_pred) if t == "leaf_roll" and p == "leaf_roll")
                / max(1, sum(1 for t in y_true if t == "leaf_roll"))
            ),
            "healthy_recall": float(
                sum(1 for t, p in zip(y_true, soft_pred) if t == "healthy" and p == "healthy")
                / max(1, sum(1 for t in y_true if t == "healthy"))
            ),
            "prediction_counts": dict(Counter(soft_pred)),
        },
        "binary_hard_vote": {
            "accuracy": float(accuracy_score(y_true, vote_pred)),
            "macro_f1": float(
                f1_score(y_true, vote_pred, labels=list(FIELD_CLASSES), average="macro", zero_division=0)
            ),
        },
        "canopy_auroc_binary": safe_auc(binary_truth, canopy_scores),
        "canopy_best_operating_point": best_operating_point(binary_truth, canopy_scores),
        "crop_auroc_binary": safe_auc(crop_truth, crop_scores),
        "mean_canopy_score_healthy": float(
            np.mean([s for s, t in zip(canopy_scores, binary_truth) if t == 0]) if 0 in binary_truth else 0.0
        ),
        "mean_canopy_score_leaf_roll": float(
            np.mean([s for s, t in zip(canopy_scores, binary_truth) if t == 1]) if 1 in binary_truth else 0.0
        ),
        "crop_accuracy_3class_vs_weak_label": float(crop_correct_3class / crop_total) if crop_total else 0.0,
        "mean_crop_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "confident_crop_rate": float(np.mean(used_flags)) if used_flags else 0.0,
        "crop_prediction_counts": dict(
            Counter(pred.predicted_class for preds in predictions_by_canopy for pred in preds)
        ),
    }


def cohort_baselines(records: list[CanopyRecord]) -> dict:
    counts = Counter(record.true_label for record in records)
    total = sum(counts.values())
    majority = max(counts.values()) / total if total else 0.0
    return {
        "n": total,
        "label_counts": dict(counts),
        "majority_class_accuracy": float(majority),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def file_fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return {
        "path": str(path),
        "exists": True,
        "sha256_12": digest,
        "modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "bytes": path.stat().st_size,
    }


def run_arms(
    records: list[CanopyRecord],
    segmenter,
    classifiers: dict[str, object],
    skip_audit: bool = False,
    overlay_stems: set[str] | None = None,
) -> list[tuple[str, Image.Image]]:
    overlay_tiles: list[tuple[str, Image.Image]] = []
    for index, record in enumerate(records, start=1):
        started = time.time()
        with Image.open(record.canopy_path) as handle:
            original = ImageOps.exif_transpose(handle).convert("RGB")

        polygons = [p for p in load_generic_polygon_json(Path(record.annotation_path)) if len(p) >= 3]
        oracle_instances: list[LeafInstance] = []
        for polygon in polygons:
            result = oracle_instance_from_polygon(original, polygon)
            if result is None:
                continue
            instance, reject_reason = result
            oracle_instances.append(instance)
            if reject_reason:
                record.oracle_gate_rejections.append(reject_reason)

        segmenter_instances, rejections = segment_image_with_audit(original, model=segmenter)
        record.segmenter_rejections = [rejection.reason for rejection in rejections]

        canopy_instances = [whole_canopy_instance(original)]

        instances_by_arm = {
            "oracle": oracle_instances,
            "segmenter": segmenter_instances,
            "canopy": canopy_instances,
        }
        record.crop_counts = {arm: len(items) for arm, items in instances_by_arm.items()}

        for classifier_key, classifier in classifiers.items():
            record.arm_predictions[classifier_key] = {
                arm: classify_instances(items, classifier) for arm, items in instances_by_arm.items()
            }

        if not skip_audit:
            canvas_size, scale = audit_canvas_geometry(original.size)
            human_masks = human_masks_on_canvas(polygons, canvas_size, scale)
            crop_masks = [
                instance_mask_on_canvas(instance, canvas_size, scale) for instance in segmenter_instances
            ]
            record.audit = audit_segmentation(human_masks, crop_masks)

        if overlay_stems and record.stem in overlay_stems:
            label = f"{record.true_label[:9]} {record.field_split[:4]} h={len(polygons)} s={len(segmenter_instances)}"
            overlay_tiles.append((label, render_overlay(original, polygons, segmenter_instances)))

        elapsed = time.time() - started
        print(
            f"  [{index}/{len(records)}] {record.stem} "
            f"({record.true_label}, {record.field_split}) "
            f"oracle={len(oracle_instances)} segmenter={len(segmenter_instances)} "
            f"{elapsed:.1f}s",
            flush=True,
        )
    return overlay_tiles


def build_cohorts(records: list[CanopyRecord]) -> dict[str, list[CanopyRecord]]:
    return {
        "all_annotated": records,
        "non_adapt": [record for record in records if record.field_split in ("validation", "test")],
        "test_only": [record for record in records if record.field_split == "test"],
    }


def attribution(cohort_results: dict) -> dict:
    """Turn arm deltas into the three loss terms the experiment is meant to separate."""
    summary = {}
    for classifier_key, arms in cohort_results.items():
        oracle = arms["oracle"]
        segmenter = arms["segmenter"]
        canopy = arms["canopy"]

        def auc(arm: dict) -> float | None:
            return arm["canopy_auroc_binary"]

        def accuracy(arm: dict) -> float:
            return arm["binary_soft_vote"]["accuracy"]

        summary[classifier_key] = {
            "segmentation_loss_auroc": (
                None if auc(oracle) is None or auc(segmenter) is None else round(auc(oracle) - auc(segmenter), 4)
            ),
            "segmentation_loss_accuracy": round(accuracy(oracle) - accuracy(segmenter), 4),
            "context_loss_auroc": (
                None if auc(canopy) is None or auc(oracle) is None else round(auc(canopy) - auc(oracle), 4)
            ),
            "context_loss_accuracy": round(accuracy(canopy) - accuracy(oracle), 4),
            "classification_headroom_auroc": (None if auc(oracle) is None else round(1.0 - auc(oracle), 4)),
            "oracle_auroc": auc(oracle),
            "segmenter_auroc": auc(segmenter),
            "canopy_auroc": auc(canopy),
        }
    return summary


def decompose(
    annotation_dir: Path,
    canopy_root: Path,
    segmenter_path: Path,
    classifier_paths: dict[str, Path],
    output_path: Path,
    split_path: Path | None,
    limit: int | None,
    skip_audit: bool,
    overlay_path: Path | None = None,
    overlay_samples: int = 12,
) -> dict:
    print("Building annotated cohort...")
    records = build_cohort(annotation_dir, canopy_root, split_path)
    if limit:
        records = records[:limit]
    print(f"  {len(records)} annotated canopies, {sum(r.human_instance_count for r in records)} human leaves")

    print("Loading models...")
    segmenter = load_segmenter(segmenter_path)
    classifiers = {key: load_classifier(path) for key, path in classifier_paths.items()}

    overlay_stems: set[str] = set()
    if overlay_path is not None and overlay_samples > 0:
        # Sample evenly across each label so the sheet is not all one class.
        by_label: dict[str, list[CanopyRecord]] = {}
        for record in records:
            by_label.setdefault(record.true_label, []).append(record)
        per_label = max(1, overlay_samples // max(1, len(by_label)))
        for label_records in by_label.values():
            step = max(1, len(label_records) // per_label)
            overlay_stems.update(record.stem for record in label_records[::step][:per_label])

    print("Running arms...")
    overlay_tiles = run_arms(
        records,
        segmenter,
        classifiers,
        skip_audit=skip_audit,
        overlay_stems=overlay_stems,
    )
    if overlay_path is not None:
        write_overlay_sheet(overlay_tiles, overlay_path)

    thresholds_by_name = {
        "config_default": dict(config.PLANT_CLASS_FRACTION_THRESHOLD),
    }
    for name, results_file in (
        ("calibrated_base", config.OUTPUTS_DIR / "field_validation_results.json"),
        ("calibrated_adapted", config.OUTPUTS_DIR / "field_validation_results_adapted.json"),
    ):
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as handle:
                thresholds_by_name[name] = json.load(handle).get("thresholds", {})

    print("Scoring...")
    cohorts = build_cohorts(records)
    results_by_cohort = {}
    for cohort_name, cohort_records in cohorts.items():
        if not cohort_records:
            continue
        arms_by_classifier = {
            classifier_key: {
                arm: arm_metrics(cohort_records, classifier_key, arm, thresholds_by_name) for arm in ARMS
            }
            for classifier_key in classifiers
        }
        results_by_cohort[cohort_name] = {
            "baseline": cohort_baselines(cohort_records),
            "split_counts": dict(Counter(record.field_split for record in cohort_records)),
            "segmenter_split_counts": dict(Counter(record.segmenter_split for record in cohort_records)),
            "arms": arms_by_classifier,
            "attribution": attribution(arms_by_classifier),
            "segmentation_audit": aggregate_audit(cohort_records),
        }

    payload = {
        "experiment": "oracle_error_decomposition",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "segmenter": file_fingerprint(segmenter_path),
            "classifiers": {key: file_fingerprint(path) for key, path in classifier_paths.items()},
            "annotation_dir": str(annotation_dir),
            "split_file": str(split_path or config.FIELD_SPLIT_DIR / "canopy_splits.json"),
        },
        "caveats": [
            "Every annotated canopy was used to train or validate the current segmenter, so the "
            "segmenter arm is optimistically biased; there is no untouched annotated canopy.",
            "The annotated set was selected by segment/select_field_failures.py as pipeline "
            "failures, so it over-represents hard canopies relative to the full field set.",
            "Canopy labels are weak plant-level labels: crops from a leaf_roll canopy inherit "
            "leaf_roll even when the individual leaf is asymptomatic.",
            "The field-adapted classifier was fine-tuned on adapt-split canopies, so read the "
            "non_adapt and test_only cohorts for it.",
        ],
        "cohorts": results_by_cohort,
        "canopies": [
            {
                "canopy_path": record.canopy_path,
                "annotation_path": record.annotation_path,
                "true_label": record.true_label,
                "field_split": record.field_split,
                "segmenter_split": record.segmenter_split,
                "human_instance_count": record.human_instance_count,
                "crop_counts": record.crop_counts,
                "segmenter_rejections": dict(Counter(record.segmenter_rejections)),
                "oracle_gate_rejections": dict(Counter(record.oracle_gate_rejections)),
                "audit": record.audit,
                "predictions": {
                    classifier_key: {
                        arm: {
                            "verdict": aggregate_with_thresholds(
                                preds, thresholds_by_name.get("calibrated_adapted")
                                or dict(config.PLANT_CLASS_FRACTION_THRESHOLD)
                            ),
                            "mean_binary_leaf_roll": float(
                                np.mean([binary_leaf_roll_probability(pred) for pred in preds])
                            )
                            if preds
                            else None,
                            "crop_predictions": dict(Counter(pred.predicted_class for pred in preds)),
                        }
                        for arm, preds in arms.items()
                    }
                    for classifier_key, arms in record.arm_predictions.items()
                },
            }
            for record in records
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(json.loads(json.dumps(payload, default=str)), handle, indent=2)
    print(f"\nOracle decomposition saved to {output_path}")
    print_summary(payload)
    return payload


def print_summary(payload: dict) -> None:
    for cohort_name, cohort in payload["cohorts"].items():
        baseline = cohort["baseline"]
        print(f"\n=== {cohort_name} (n={baseline['n']}, majority={baseline['majority_class_accuracy']:.3f}) ===")
        print(f"    splits: {cohort['split_counts']}")
        for classifier_key, arms in cohort["arms"].items():
            print(f"  classifier: {classifier_key}")
            header = (
                f"    {'arm':<10}{'crops':>7}{'AUROC':>8}{'bestAcc':>9}{'bestBal':>9}"
                f"{'binAcc':>8}{'binF1':>8}{'3clsAcc':>9}{'3clsF1':>8}"
            )
            print(header)
            for arm in ARMS:
                metrics = arms[arm]
                auc = metrics["canopy_auroc_binary"]
                hard = metrics["hard_vote_3class"]["config_default"]
                best = metrics.get("canopy_best_operating_point") or {}
                print(
                    f"    {arm:<10}{metrics['n_crops']:>7}"
                    f"{(f'{auc:.3f}' if auc is not None else 'n/a'):>8}"
                    f"{best.get('accuracy', float('nan')):>9.3f}"
                    f"{best.get('balanced_accuracy', float('nan')):>9.3f}"
                    f"{metrics['binary_soft_vote']['accuracy']:>8.3f}"
                    f"{metrics['binary_soft_vote']['macro_f1']:>8.3f}"
                    f"{hard['accuracy']:>9.3f}"
                    f"{hard['macro_f1']:>8.3f}"
                )
        audit = cohort.get("segmentation_audit") or {}
        if audit:
            print(
                f"    segmentation audit: recall@0.5={audit['instance_recall_iou50']:.3f} "
                f"precision@0.5={audit['instance_precision_iou50']:.3f} "
                f"purity={audit['mean_crop_purity']:.3f} "
                f"merged_crops={audit['merged_crop_rate']:.3f} "
                f"dice={audit['mean_semantic_dice']:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle error decomposition on annotated field canopies.")
    parser.add_argument("--annotation-dir", type=Path, default=config.FIELD_SEGMENTATION_DATA_DIR / "raw")
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--segmenter-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--base-classifier-path", type=Path, default=config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME)
    parser.add_argument(
        "--adapted-classifier-path",
        type=Path,
        default=config.MODELS_DIR / "autorogue_leaf_classifier_field_adapted.keras",
    )
    parser.add_argument("--output-path", type=Path, default=config.OUTPUTS_DIR / "oracle_error_decomposition.json")
    parser.add_argument("--split-path", type=Path, default=config.FIELD_SPLIT_DIR / "canopy_splits.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--overlay-path",
        type=Path,
        default=config.OUTPUTS_DIR / "oracle_decomposition_overlays.jpg",
    )
    parser.add_argument("--overlay-samples", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier_paths = {"base": args.base_classifier_path}
    if args.adapted_classifier_path.exists():
        classifier_paths["field_adapted"] = args.adapted_classifier_path
    decompose(
        annotation_dir=args.annotation_dir,
        canopy_root=args.canopy_root,
        segmenter_path=args.segmenter_path,
        classifier_paths=classifier_paths,
        output_path=args.output_path,
        split_path=args.split_path,
        limit=args.limit,
        skip_audit=args.skip_audit,
        overlay_path=args.overlay_path,
        overlay_samples=args.overlay_samples,
    )


if __name__ == "__main__":
    main()
