"""Run the field recovery loop after classifier and segmenter models exist."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from datasets.canopy_to_leaves import segment_canopy_dataset
from domain_adapt import fine_tune_on_field_leaves
from field_splits import ADAPT_SPLIT, split_file, write_field_splits
from field_validate import validate_field_pipeline


def require_model(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. "
            "Train the classifier and segmenter before running recovery."
        )


def run_recovery(
    canopy_root: Path = config.CANOPY_DIR,
    field_root: Path = config.FIELD_LEAF_DIR,
    segmenter_path: Path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME,
    classifier_path: Path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME,
    adapted_classifier_path: Path = config.MODELS_DIR / "autorogue_leaf_classifier_field_adapted.keras",
    output_path: Path = config.OUTPUTS_DIR / "field_validation_results.json",
    clean_field_crops: bool = True,
    adapt_epochs: int = 10,
    adapt_learning_rate: float = 5e-6,
) -> dict:
    """Create splits, regenerate field crops, adapt classifier, and validate."""
    require_model(segmenter_path, "segmenter model")
    require_model(classifier_path, "base classifier model")

    write_field_splits(canopy_root=canopy_root, output_dir=config.FIELD_SPLIT_DIR)
    split_path = split_file()

    segment_canopy_dataset(
        canopy_root=canopy_root,
        output_root=field_root,
        model_path=segmenter_path,
        clean_output=clean_field_crops,
    )

    fine_tune_on_field_leaves(
        base_model_path=classifier_path,
        field_root=field_root,
        output_path=adapted_classifier_path,
        epochs=adapt_epochs,
        learning_rate=adapt_learning_rate,
        field_split=ADAPT_SPLIT,
        split_path=split_path,
    )

    return validate_field_pipeline(
        canopy_root=canopy_root,
        segmenter_path=segmenter_path,
        classifier_path=adapted_classifier_path,
        output_path=output_path,
        split_path=split_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Autorogue field recovery loop.")
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--field-root", type=Path, default=config.FIELD_LEAF_DIR)
    parser.add_argument("--segmenter-path", type=Path, default=config.MODELS_DIR / config.SEGMENTER_MODEL_NAME)
    parser.add_argument("--classifier-path", type=Path, default=config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME)
    parser.add_argument(
        "--adapted-classifier-path",
        type=Path,
        default=config.MODELS_DIR / "autorogue_leaf_classifier_field_adapted.keras",
    )
    parser.add_argument("--output-path", type=Path, default=config.OUTPUTS_DIR / "field_validation_results.json")
    parser.add_argument("--keep-old-field-crops", action="store_true")
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--adapt-learning-rate", type=float, default=5e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_recovery(
        canopy_root=args.canopy_root,
        field_root=args.field_root,
        segmenter_path=args.segmenter_path,
        classifier_path=args.classifier_path,
        adapted_classifier_path=args.adapted_classifier_path,
        output_path=args.output_path,
        clean_field_crops=not args.keep_old_field_crops,
        adapt_epochs=args.adapt_epochs,
        adapt_learning_rate=args.adapt_learning_rate,
    )


if __name__ == "__main__":
    main()
