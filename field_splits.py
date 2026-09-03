"""Canopy-level splits for weak field adaptation and validation."""

from __future__ import annotations

import json
import random
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import config
from datasets.ingest import IMAGE_EXTENSIONS

SPLIT_FILENAME = "canopy_splits.json"
ADAPT_SPLIT = "adapt"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
QUARANTINE_SPLIT = "quarantine"

# Only `adapt` may be fitted on. Quarantined canopies are near-duplicates of test
# canopies and are excluded from everything.
TRAINABLE_SPLITS = frozenset({ADAPT_SPLIT})
HELD_OUT_SPLITS = frozenset({VALIDATION_SPLIT, TEST_SPLIT, QUARANTINE_SPLIT})


@dataclass(frozen=True)
class CanopyExample:
    image_path: str
    true_label: str
    source_class: str
    split: str
    # Populated by field_holdout.py. Legacy canopy_splits.json records omit them.
    group_id: str = ""
    plot_id: str = ""
    session_id: str = ""
    captured_at: str = ""


def list_canopy_examples(canopy_root: Path = config.CANOPY_DIR) -> list[CanopyExample]:
    """List plant-labelled canopy images without assigning a split."""
    source_map = config.SOURCE_CLASS_MAP["canopy_weak"]
    examples: list[CanopyExample] = []
    for source_class, unified_class in source_map.items():
        if unified_class == "ignore":
            continue
        class_dir = canopy_root / source_class
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                examples.append(
                    CanopyExample(
                        image_path=str(image_path),
                        true_label=unified_class,
                        source_class=source_class,
                        split="",
                    )
                )
    return examples


def assign_stratified_splits(
    examples: list[CanopyExample],
    seed: int = config.SEED,
    val_fraction: float = config.FIELD_VAL_FRACTION,
    test_fraction: float = config.FIELD_TEST_FRACTION,
) -> list[CanopyExample]:
    """Assign adapt/validation/test splits within each field class."""
    grouped: dict[str, list[CanopyExample]] = {}
    for example in examples:
        grouped.setdefault(example.true_label, []).append(example)

    split_examples: list[CanopyExample] = []
    rng = random.Random(seed)
    for label in sorted(grouped):
        class_examples = grouped[label][:]
        rng.shuffle(class_examples)
        n_total = len(class_examples)
        n_test = max(1, int(n_total * test_fraction)) if n_total else 0
        n_val = max(1, int(n_total * val_fraction)) if n_total - n_test > 1 else 0

        for index, example in enumerate(class_examples):
            if index < n_val:
                split = VALIDATION_SPLIT
            elif index < n_val + n_test:
                split = TEST_SPLIT
            else:
                split = ADAPT_SPLIT
            split_examples.append(
                CanopyExample(
                    image_path=example.image_path,
                    true_label=example.true_label,
                    source_class=example.source_class,
                    split=split,
                )
            )
    split_examples.sort(key=lambda item: (item.split, item.true_label, item.image_path))
    return split_examples


def split_file(output_dir: Path = config.FIELD_SPLIT_DIR) -> Path:
    return output_dir / SPLIT_FILENAME


def write_field_splits(
    canopy_root: Path = config.CANOPY_DIR,
    output_dir: Path = config.FIELD_SPLIT_DIR,
    seed: int = config.SEED,
) -> list[CanopyExample]:
    """Create and persist canopy-level splits."""
    examples = assign_stratified_splits(list_canopy_examples(canopy_root), seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(split_file(output_dir), "w", encoding="utf-8") as handle:
        json.dump([asdict(example) for example in examples], handle, indent=2)
    return examples


def load_field_splits(
    split_path: Path | None = None,
    canopy_root: Path = config.CANOPY_DIR,
    create_if_missing: bool = True,
) -> list[CanopyExample]:
    """Load persisted splits, optionally creating them from canopy folders."""
    if split_path is None:
        split_path = split_file()
    if not split_path.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"No field split file found at {split_path}")
        if split_path.name != SPLIT_FILENAME:
            # Only the legacy ungrouped split can be regenerated from folder contents.
            # Silently rebuilding a grouped split here would produce a leaked one
            # under a filename that promises the opposite.
            raise FileNotFoundError(
                f"No field split file at {split_path}. Run `python field_holdout.py` to "
                f"build the capture-grouped split before training or evaluating."
            )
        return write_field_splits(canopy_root=canopy_root, output_dir=split_path.parent)

    with open(split_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    return [CanopyExample(**record) for record in records]


def examples_for_split(
    split_name: str,
    split_path: Path | None = None,
    canopy_root: Path = config.CANOPY_DIR,
) -> list[tuple[Path, str]]:
    """Return `(image_path, true_label)` examples for one split."""
    examples = load_field_splits(split_path=split_path, canopy_root=canopy_root)
    return [
        (Path(example.image_path), example.true_label)
        for example in examples
        if example.split == split_name
    ]


def split_lookup(
    split_path: Path | None = None,
    canopy_root: Path = config.CANOPY_DIR,
) -> dict[str, str]:
    """Map canopy image paths to split names."""
    lookup: dict[str, str] = {}
    for example in load_field_splits(split_path=split_path, canopy_root=canopy_root):
        path = Path(example.image_path)
        lookup[str(path)] = example.split
        lookup[path.as_posix()] = example.split
        lookup[str(path).replace("/", "\\")] = example.split
        lookup[str(path).replace("\\", "/")] = example.split
    return lookup


def excluded_canopy_stems(
    split_path: Path | None = None,
    canopy_root: Path = config.CANOPY_DIR,
    missing_ok: bool = True,
) -> set[str]:
    """Filename stems of canopies that no model may be fitted on.

    Anything outside the `adapt` split is held out, so this is the guard that
    keeps test canopies out of segmenter training, crop generation, and
    annotation selection. Returns an empty set when no split file exists so the
    legacy ungrouped workflow keeps running.
    """
    if split_path is None:
        split_path = config.UNTOUCHED_SPLIT_PATH
    if not Path(split_path).exists():
        if missing_ok:
            return set()
        raise FileNotFoundError(f"No field split file found at {split_path}")

    examples = load_field_splits(
        split_path=Path(split_path), canopy_root=canopy_root, create_if_missing=False
    )
    return {
        Path(example.image_path).stem
        for example in examples
        if example.split not in TRAINABLE_SPLITS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canopy-level field splits.")
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.FIELD_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = write_field_splits(args.canopy_root, args.output_dir, args.seed)
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.split] = counts.get(example.split, 0) + 1
    print(f"Wrote {len(examples)} canopy split records to {split_file(args.output_dir)}")
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()
