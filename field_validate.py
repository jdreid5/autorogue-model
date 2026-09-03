"""Field-only end-to-end validation for canopy images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config
from field_splits import TEST_SPLIT, VALIDATION_SPLIT, examples_for_split, load_field_splits
from pipeline.infer_canopy import (
    LeafPrediction,
    load_classifier,
    predict_canopy,
    prediction_to_json,
)
from segment.infer import load_segmenter

ALL_PREDICTION_LABELS = config.CLASSES + ["uncertain"]


def aggregate_with_thresholds(
    leaf_predictions: list[LeafPrediction],
    thresholds: dict[str, float],
) -> str:
    used = [pred for pred in leaf_predictions if pred.used_for_aggregation]
    if not used:
        return "uncertain"

    fractions = {
        class_name: sum(1 for pred in used if pred.predicted_class == class_name) / len(used)
        for class_name in config.CLASSES
    }
    for class_name, threshold in thresholds.items():
        if fractions.get(class_name, 0.0) >= threshold:
            return class_name
    if fractions["healthy"] >= config.PLANT_HEALTHY_MIN_CONFIDENCE:
        return "healthy"
    return max(fractions, key=fractions.get)


def calibrate_thresholds(records: list[dict]) -> dict[str, float]:
    thresholds = dict(config.PLANT_CLASS_FRACTION_THRESHOLD)
    candidate_values = np.arange(0.05, 0.55, 0.05)

    for disease_class in thresholds:
        if not any(record["true_label"] == disease_class for record in records):
            continue
        best_threshold = thresholds[disease_class]
        best_score = -1.0
        for candidate in candidate_values:
            trial_thresholds = dict(thresholds)
            trial_thresholds[disease_class] = float(candidate)
            y_true = [record["true_label"] for record in records]
            y_pred = [
                aggregate_with_thresholds(record["leaf_predictions"], trial_thresholds)
                for record in records
            ]
            score = f1_score(y_true, y_pred, labels=config.CLASSES, average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(candidate)
        thresholds[disease_class] = best_threshold
    return thresholds


def run_predictions(examples: list[tuple[Path, str]], segmenter, classifier) -> list[dict]:
    records = []
    for image_path, true_label in examples:
        prediction = predict_canopy(image_path, segmenter=segmenter, classifier=classifier)
        prediction_json = prediction_to_json(prediction)
        records.append(
            {
                "image_path": str(image_path),
                "true_label": true_label,
                "prediction": prediction_json,
                "leaf_predictions": prediction.leaf_predictions,
            }
        )
    return records


def diagnostics(records: list[dict], y_pred: list[str]) -> dict:
    """Summarize crop coverage and uncertainty, including zero-crop failures."""
    n_records = len(records)
    leaf_counts = [int(record["prediction"]["leaf_count"]) for record in records]
    used_leaf_counts = [int(record["prediction"]["used_leaf_count"]) for record in records]
    confidences = [float(record["prediction"]["confidence"]) for record in records]
    prediction_counts = Counter(y_pred)
    true_counts = Counter(record["true_label"] for record in records)

    return {
        "prediction_counts": dict(prediction_counts),
        "true_counts": dict(true_counts),
        "uncertain_count": int(prediction_counts.get("uncertain", 0)),
        "uncertain_rate": float(prediction_counts.get("uncertain", 0) / n_records) if n_records else 0.0,
        "zero_crop_count": int(sum(1 for count in leaf_counts if count == 0)),
        "zero_crop_rate": float(sum(1 for count in leaf_counts if count == 0) / n_records) if n_records else 0.0,
        "zero_confident_crop_count": int(sum(1 for count in used_leaf_counts if count == 0)),
        "zero_confident_crop_rate": float(sum(1 for count in used_leaf_counts if count == 0) / n_records)
        if n_records
        else 0.0,
        "mean_leaf_count": float(np.mean(leaf_counts)) if leaf_counts else 0.0,
        "median_leaf_count": float(np.median(leaf_counts)) if leaf_counts else 0.0,
        "mean_used_leaf_count": float(np.mean(used_leaf_counts)) if used_leaf_counts else 0.0,
        "median_used_leaf_count": float(np.median(used_leaf_counts)) if used_leaf_counts else 0.0,
        "mean_verdict_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def summarize(records: list[dict], thresholds: dict[str, float]) -> dict:
    y_true = [record["true_label"] for record in records]
    y_pred = [
        aggregate_with_thresholds(record["leaf_predictions"], thresholds)
        for record in records
    ]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=config.CLASSES, average="macro", zero_division=0))
        if y_true
        else 0.0,
        "macro_f1_with_uncertain": float(
            f1_score(y_true, y_pred, labels=ALL_PREDICTION_LABELS, average="macro", zero_division=0)
        )
        if y_true
        else 0.0,
        "labels": ALL_PREDICTION_LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ALL_PREDICTION_LABELS).tolist()
        if y_true
        else [],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=ALL_PREDICTION_LABELS,
            zero_division=0,
            output_dict=True,
        )
        if y_true
        else {},
        "diagnostics": diagnostics(records, y_pred),
        "predictions": [
            {
                "image_path": record["image_path"],
                "true_label": true,
                "predicted_label": pred,
                "verdict_confidence": float(record["prediction"]["confidence"]),
                "leaf_count": int(record["prediction"]["leaf_count"]),
                "used_leaf_count": int(record["prediction"]["used_leaf_count"]),
                "class_fractions": record["prediction"]["class_fractions"],
            }
            for record, true, pred in zip(records, y_true, y_pred)
        ],
    }


def validate_field_pipeline(
    canopy_root: Path = config.CANOPY_DIR,
    segmenter_path: Path | None = None,
    classifier_path: Path | None = None,
    output_path: Path | None = None,
    split_path: Path | None = None,
) -> dict:
    if output_path is None:
        output_path = config.OUTPUTS_DIR / "field_validation_results.json"
    if segmenter_path is None:
        segmenter_path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME
    if classifier_path is None:
        classifier_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME

    split_records = load_field_splits(split_path=split_path, canopy_root=canopy_root)
    val_examples = examples_for_split(VALIDATION_SPLIT, split_path=split_path, canopy_root=canopy_root)
    test_examples = examples_for_split(TEST_SPLIT, split_path=split_path, canopy_root=canopy_root)
    segmenter = load_segmenter(segmenter_path)
    classifier = load_classifier(classifier_path)

    val_records = run_predictions(val_examples, segmenter, classifier)
    thresholds = calibrate_thresholds(val_records)
    test_records = run_predictions(test_examples, segmenter, classifier)

    results = {
        "split_file": str(split_path or config.UNTOUCHED_SPLIT_PATH),
        "split_counts": dict(Counter(record.split for record in split_records)),
        "thresholds": thresholds,
        "validation": summarize(val_records, thresholds),
        "test": summarize(test_records, thresholds),
        "n_validation": len(val_records),
        "n_test": len(test_records),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(results, default=str))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    print(f"Field validation results saved to {output_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate full pipeline on held-out canopy images.")
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--segmenter-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--classifier-path", type=Path, default=config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME)
    parser.add_argument("--output-path", type=Path, default=config.OUTPUTS_DIR / "field_validation_results.json")
    parser.add_argument("--split-path", type=Path, default=config.UNTOUCHED_SPLIT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_field_pipeline(
        args.canopy_root,
        args.segmenter_path,
        args.classifier_path,
        args.output_path,
        args.split_path,
    )


if __name__ == "__main__":
    main()
