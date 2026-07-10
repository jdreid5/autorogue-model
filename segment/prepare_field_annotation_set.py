"""Prepare selected field canopy images for manual segmentation annotation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import config


def safe_annotation_name(record: dict) -> str:
    source_class = record["source_class"].replace("-", "_")
    image_path = Path(record["canopy_path"])
    return f"{source_class}__{image_path.stem}{image_path.suffix.lower()}"


def prepare_annotation_set(candidates_path: Path, output_dir: Path, clean: bool = False) -> list[dict]:
    with open(candidates_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    candidates = payload.get("candidates", [])
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for candidate in candidates:
        source_path = Path(candidate["canopy_path"])
        if not source_path.exists():
            continue
        output_name = safe_annotation_name(candidate)
        output_path = output_dir / output_name
        shutil.copy2(source_path, output_path)
        records.append(
            {
                **candidate,
                "annotation_image_path": str(output_path),
                "expected_annotation_path": str(output_path.with_suffix(".json")),
            }
        )

    manifest_path = output_dir / "annotation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"count": len(records), "images": records}, handle, indent=2)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy selected field failures into an annotation folder.")
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=config.OUTPUTS_DIR / "field_segmentation_annotation_candidates.json",
    )
    parser.add_argument("--output-dir", type=Path, default=config.FIELD_SEGMENTATION_DATA_DIR / "raw")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = prepare_annotation_set(args.candidates_path, args.output_dir, clean=args.clean)
    print(f"Prepared {len(records)} images for manual annotation in {args.output_dir}")
    print(f"Annotate each image with same-stem JSON polygons, then run:")
    print(f"py -3.13 -m segment.data --raw-root {args.output_dir} --output-root {config.FIELD_SEGMENTATION_DATA_DIR}")


if __name__ == "__main__":
    main()
