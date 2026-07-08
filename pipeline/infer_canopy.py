"""End-to-end canopy inference: segment -> classify leaves -> aggregate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np
from PIL import Image

import config
import preprocess
from segment.infer import LeafInstance, load_segmenter, prepare_leaf_for_classifier, segment_image


@dataclass
class LeafPrediction:
    bbox: tuple[int, int, int, int]
    area: int
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float
    used_for_aggregation: bool


@dataclass
class CanopyPrediction:
    image_path: str
    verdict: str
    confidence: float
    leaf_count: int
    used_leaf_count: int
    class_fractions: dict[str, float]
    leaf_predictions: list[LeafPrediction]


def load_classifier(model_path: Path | None = None) -> keras.Model:
    if model_path is None:
        model_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    return keras.models.load_model(model_path, compile=False)


def prepare_leaf_batch(instances: list[LeafInstance]) -> np.ndarray:
    images = []
    for instance in instances:
        crop = prepare_leaf_for_classifier(instance.crop)
        images.append(np.asarray(crop, dtype=np.float32))
    if not images:
        return np.empty((0, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
    batch = np.asarray(images)
    return np.asarray(preprocess.preprocess_for_mobilenet(batch))


def classify_instances(
    instances: list[LeafInstance],
    classifier: keras.Model,
) -> list[LeafPrediction]:
    batch = prepare_leaf_batch(instances)
    if len(batch) == 0:
        return []
    probabilities = classifier.predict(batch, verbose=0)
    predictions: list[LeafPrediction] = []
    for instance, probs in zip(instances, probabilities):
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        predicted_class = config.CLASSES[class_idx]
        predictions.append(
            LeafPrediction(
                bbox=instance.bbox,
                area=instance.area,
                probabilities={class_name: float(probs[idx]) for idx, class_name in enumerate(config.CLASSES)},
                predicted_class=predicted_class,
                confidence=confidence,
                used_for_aggregation=confidence >= config.PER_LEAF_CONFIDENCE_THRESHOLD,
            )
        )
    return predictions


def aggregate_leaf_predictions(predictions: list[LeafPrediction]) -> tuple[str, float, dict[str, float], int]:
    used = [pred for pred in predictions if pred.used_for_aggregation]
    if not used:
        return "uncertain", 0.0, {class_name: 0.0 for class_name in config.CLASSES}, 0

    fractions = {}
    for class_name in config.CLASSES:
        fractions[class_name] = sum(1 for pred in used if pred.predicted_class == class_name) / len(used)

    disease_candidates = []
    for class_name, threshold in config.PLANT_CLASS_FRACTION_THRESHOLD.items():
        fraction = fractions.get(class_name, 0.0)
        if fraction >= threshold:
            disease_candidates.append((class_name, fraction))

    if disease_candidates:
        verdict, confidence = max(disease_candidates, key=lambda item: item[1])
        return verdict, confidence, fractions, len(used)

    healthy_fraction = fractions.get("healthy", 0.0)
    if healthy_fraction >= config.PLANT_HEALTHY_MIN_CONFIDENCE:
        return "healthy", healthy_fraction, fractions, len(used)

    verdict = max(fractions, key=fractions.get)
    return verdict, fractions[verdict], fractions, len(used)


def predict_canopy(
    image_path: Path,
    segmenter: keras.Model | None = None,
    classifier: keras.Model | None = None,
    segmenter_path: Path | None = None,
    classifier_path: Path | None = None,
) -> CanopyPrediction:
    if segmenter is None:
        segmenter = load_segmenter(segmenter_path)
    if classifier is None:
        classifier = load_classifier(classifier_path)

    with Image.open(image_path) as image:
        instances = segment_image(image, model=segmenter)

    leaf_predictions = classify_instances(instances, classifier)
    verdict, confidence, fractions, used_count = aggregate_leaf_predictions(leaf_predictions)
    return CanopyPrediction(
        image_path=str(image_path),
        verdict=verdict,
        confidence=float(confidence),
        leaf_count=len(leaf_predictions),
        used_leaf_count=used_count,
        class_fractions=fractions,
        leaf_predictions=leaf_predictions,
    )


def prediction_to_json(prediction: CanopyPrediction) -> dict:
    data = asdict(prediction)
    data["leaf_predictions"] = [asdict(pred) for pred in prediction.leaf_predictions]
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end canopy inference.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--segmenter-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--classifier-path", type=Path, default=config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = predict_canopy(
        image_path=args.image_path,
        segmenter_path=args.segmenter_path,
        classifier_path=args.classifier_path,
    )
    data = prediction_to_json(prediction)
    print(json.dumps(data, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)


if __name__ == "__main__":
    main()
