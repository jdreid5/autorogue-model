#!/usr/bin/env python3
"""
Create source-aware train/val/test splits for Autorogue image datasets.

The splitter expects class subdirectories under --input-root and writes
split/class subdirectories under --output-root. Filenames produced by
datasets/ingest.py include the source prefix; this script groups by that prefix
so images from the same source/class block do not leak between splits.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import config
from datasets.ingest import IMAGE_EXTENSIONS

DEFAULT_SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def source_group_key(path: Path) -> str:
    """Group augmented siblings by source, class, and original image identity."""
    parts = path.name.split("__")
    if len(parts) >= 3:
        source = parts[0]
        source_class = parts[1]
        original_stem = "__".join(parts[2:]).rsplit(".", 1)[0]

        # Common augmentation/collision suffixes should stay with the original.
        original_stem = re.sub(r"_jpg\\.rf\\.[0-9a-fA-F]+$", "_jpg", original_stem)
        original_stem = re.sub(r"_(?:flip|rot|rotate|shear|brightness|noise|aug)[_-]?\\d*$", "", original_stem)
        original_stem = re.sub(r"_\\d+$", "", original_stem)
        return f"{source}__{source_class}__{original_stem}"
    return path.stem


def copy_split(groups: list[list[Path]], output_root: Path, class_name: str, split: str) -> int:
    count = 0
    target_dir = output_root / split / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        for src in group:
            shutil.copy2(src, target_dir / src.name)
            count += 1
    return count


def split_class(
    input_root: Path,
    output_root: Path,
    class_name: str,
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, int]:
    class_dir = input_root / class_name
    if not class_dir.exists():
        print(f"Skipping missing class directory: {class_dir}")
        return {split: 0 for split in split_ratios}

    buckets: dict[str, list[Path]] = defaultdict(list)
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            buckets[source_group_key(path)].append(path)

    groups = list(buckets.values())
    random.Random(seed).shuffle(groups)

    n_groups = len(groups)
    n_train = int(n_groups * split_ratios["train"])
    n_val = int(n_groups * split_ratios["val"])

    split_groups = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }

    counts = {}
    for split, group_list in split_groups.items():
        counts[split] = copy_split(group_list, output_root, class_name, split)
    return counts


def split_dataset(
    input_root: Path = config.HARMONIZED_LEAF_DIR,
    output_root: Path = config.SPLIT_DIR,
    split_ratios: dict[str, float] = DEFAULT_SPLITS,
    seed: int = config.SEED,
) -> dict[str, dict[str, int]]:
    if round(sum(split_ratios.values()), 6) != 1:
        raise ValueError("Split ratios must sum to 1.0")

    if output_root.exists():
        shutil.rmtree(output_root)

    results = {}
    for class_name in config.CLASSES:
        results[class_name] = split_class(input_root, output_root, class_name, split_ratios, seed)

    print(f"Done -> {output_root}")
    for class_name, counts in results.items():
        print(f"{class_name}: {counts}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Autorogue image data.")
    parser.add_argument("--input-root", type=Path, default=config.HARMONIZED_LEAF_DIR)
    parser.add_argument("--output-root", type=Path, default=config.SPLIT_DIR)
    parser.add_argument("--train", type=float, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--val", type=float, default=DEFAULT_SPLITS["val"])
    parser.add_argument("--test", type=float, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        split_ratios={"train": args.train, "val": args.val, "test": args.test},
        seed=args.seed,
    )


if __name__ == "__main__":
    main()