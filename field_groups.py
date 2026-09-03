"""Capture-metadata grouping keys for field canopies.

`utility scripts/background_crop.py` re-encoded every canopy without EXIF, so
`data/cropped-images` carries no capture metadata at all. The pristine originals
in `data/healthy-russets` and `data/leaf-roll-russets` still do, and filenames
are a stable join key between the two trees, so session, plot, burst, and plant
identity can be recovered without touching pixels.

The grouping keys produced here are what makes a leak-free split possible: the
capture protocol was to photograph one plant many times in quick succession, so
image-level splitting scatters near-duplicate frames of a single plant across
train and test.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

import config
from datasets.ingest import IMAGE_EXTENSIONS

GROUPS_FILENAME = "canopy_groups.json"

# Originals keyed by the source-class folder name shared with config.CANOPY_DIR.
ORIGINAL_CANOPY_DIRS = {
    "healthy-russets": config.DATA_ROOT / "healthy-russets",
    "leaf-roll-russets": config.DATA_ROOT / "leaf-roll-russets",
}

EXIF_DATETIME_ORIGINAL = 36867
EXIF_SUBSEC_TIME_ORIGINAL = 37521
EXIF_OFFSET_TIME_ORIGINAL = 36880
EXIF_MAKE = 271
EXIF_MODEL = 272

GPS_LATITUDE_REF = 1
GPS_LATITUDE = 2
GPS_LONGITUDE_REF = 3
GPS_LONGITUDE = 4
GPS_ALTITUDE = 6
GPS_IMG_DIRECTION = 17

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class CanopyCapture:
    """One canopy image plus the capture metadata recovered from its original."""

    image_path: str
    original_path: str
    source_class: str
    true_label: str
    captured_at: str
    capture_timestamp: float
    device: str
    latitude: float | None
    longitude: float | None
    altitude: float | None
    bearing: float | None
    session_id: str
    plot_id: str
    burst_id: str
    plant_id: str
    group_id: str
    burst_position: int
    burst_size: int


def _rational_to_degrees(value) -> float:
    degrees, minutes, seconds = (float(part) for part in value)
    return degrees + minutes / 60.0 + seconds / 3600.0


def _haversine_metres(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def read_capture_metadata(image_path: Path) -> dict:
    """Read capture time, device, and GPS from an image's EXIF."""
    with Image.open(image_path) as image:
        exif = image.getexif()
        base = dict(exif)
        exif_ifd = dict(exif.get_ifd(ExifTags.IFD.Exif))
        gps_ifd = dict(exif.get_ifd(ExifTags.IFD.GPSInfo))

    captured_at = exif_ifd.get(EXIF_DATETIME_ORIGINAL)
    if not captured_at:
        raise ValueError(f"{image_path} has no EXIF DateTimeOriginal; cannot group it")

    subsec = str(exif_ifd.get(EXIF_SUBSEC_TIME_ORIGINAL, "0") or "0")
    fractional = float(f"0.{subsec}") if subsec.isdigit() else 0.0
    naive = datetime.strptime(captured_at, "%Y:%m:%d %H:%M:%S")

    latitude = longitude = None
    if GPS_LATITUDE in gps_ifd and GPS_LONGITUDE in gps_ifd:
        latitude = _rational_to_degrees(gps_ifd[GPS_LATITUDE])
        longitude = _rational_to_degrees(gps_ifd[GPS_LONGITUDE])
        if str(gps_ifd.get(GPS_LATITUDE_REF, "N")).upper().startswith("S"):
            latitude = -latitude
        if str(gps_ifd.get(GPS_LONGITUDE_REF, "E")).upper().startswith("W"):
            longitude = -longitude

    make = str(base.get(EXIF_MAKE, "")).strip()
    model = str(base.get(EXIF_MODEL, "")).strip()

    return {
        "captured_at": naive.isoformat(),
        "capture_timestamp": naive.timestamp() + fractional,
        "capture_date": naive.strftime("%Y-%m-%d"),
        "utc_offset": exif_ifd.get(EXIF_OFFSET_TIME_ORIGINAL),
        "device": " ".join(part for part in (make, model) if part) or "unknown",
        "latitude": latitude,
        "longitude": longitude,
        "altitude": float(gps_ifd[GPS_ALTITUDE]) if GPS_ALTITUDE in gps_ifd else None,
        "bearing": float(gps_ifd[GPS_IMG_DIRECTION]) if GPS_IMG_DIRECTION in gps_ifd else None,
    }


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")


def cluster_plots(
    records: list[dict],
    radius_metres: float,
) -> dict[tuple[float, float], int]:
    """Single-linkage cluster distinct GPS fixes into plots.

    The phone reports one fix per stationary position, so distinct fixes are a
    direct proxy for where the operator stood. Single linkage merges fixes that
    a walking operator would consider the same plot.
    """
    fixes = sorted({
        (round(record["latitude"], 6), round(record["longitude"], 6))
        for record in records
        if record["latitude"] is not None and record["longitude"] is not None
    })
    parent = list(range(len(fixes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(fixes)):
        for j in range(i + 1, len(fixes)):
            if _haversine_metres(fixes[i], fixes[j]) <= radius_metres:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    # Number plots by first appearance in time so ids are stable and readable.
    order: dict[int, int] = {}
    for record in sorted(records, key=lambda item: item["capture_timestamp"]):
        if record["latitude"] is None or record["longitude"] is None:
            continue
        key = (round(record["latitude"], 6), round(record["longitude"], 6))
        root = find(fixes.index(key))
        if root not in order:
            order[root] = len(order)

    return {fix: order[find(index)] for index, fix in enumerate(fixes)}


def build_capture_groups(
    canopy_root: Path = config.CANOPY_DIR,
    original_dirs: dict[str, Path] | None = None,
    burst_gap_seconds: float = config.FIELD_BURST_GAP_SECONDS,
    plot_radius_metres: float = config.FIELD_PLOT_RADIUS_METRES,
) -> tuple[list[CanopyCapture], dict]:
    """Recover capture metadata for every canopy and derive its grouping keys."""
    if original_dirs is None:
        original_dirs = ORIGINAL_CANOPY_DIRS

    source_map = config.SOURCE_CLASS_MAP["canopy_weak"]
    raw: list[dict] = []
    missing_originals: list[str] = []

    for source_class, unified_class in source_map.items():
        if unified_class == "ignore":
            continue
        class_dir = canopy_root / source_class
        original_dir = original_dirs.get(source_class)
        if not class_dir.exists():
            continue
        if original_dir is None or not original_dir.exists():
            raise FileNotFoundError(
                f"No original-image directory for {source_class!r}; capture metadata "
                f"cannot be recovered because {canopy_root} has no EXIF"
            )

        originals = {path.stem: path for path in original_dir.rglob("*") if path.is_file()}
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            original = originals.get(image_path.stem)
            if original is None:
                missing_originals.append(str(image_path))
                continue
            metadata = read_capture_metadata(original)
            metadata.update(
                image_path=str(image_path),
                original_path=str(original),
                source_class=source_class,
                true_label=unified_class,
            )
            raw.append(metadata)

    if missing_originals:
        raise FileNotFoundError(
            f"{len(missing_originals)} canopies have no original with EXIF, so they cannot be "
            f"grouped: {missing_originals[:5]}"
        )

    plot_of_fix = cluster_plots(raw, plot_radius_metres)

    # A session is one operator run: a single device on a single date.
    for record in raw:
        record["session_id"] = f"{record['capture_date']}__{_slug(record['device'])}"
        if record["latitude"] is None or record["longitude"] is None:
            record["plot_index"] = None
        else:
            key = (round(record["latitude"], 6), round(record["longitude"], 6))
            record["plot_index"] = plot_of_fix[key]

    captures: list[CanopyCapture] = []
    # Bursts are time-contiguous runs within one class. The capture protocol was
    # to shoot one plant repeatedly before moving on, so a burst is the best
    # available proxy for plant identity.
    by_class: dict[tuple[str, str], list[dict]] = {}
    for record in raw:
        by_class.setdefault((record["session_id"], record["source_class"]), []).append(record)

    for (session_id, source_class), records in sorted(by_class.items()):
        records.sort(key=lambda item: item["capture_timestamp"])
        bursts: list[list[dict]] = [[records[0]]]
        for previous, current in zip(records, records[1:]):
            if current["capture_timestamp"] - previous["capture_timestamp"] > burst_gap_seconds:
                bursts.append([current])
            else:
                bursts[-1].append(current)

        for burst_index, burst in enumerate(bursts):
            plot_indices = [r["plot_index"] for r in burst if r["plot_index"] is not None]
            plot_index = max(set(plot_indices), key=plot_indices.count) if plot_indices else None
            plot_id = f"{session_id}__plot{plot_index:02d}" if plot_index is not None else f"{session_id}__plot-unknown"
            burst_id = f"{session_id}__{_slug(source_class)}__burst{burst_index:03d}"

            for position, record in enumerate(burst):
                captures.append(
                    CanopyCapture(
                        image_path=record["image_path"],
                        original_path=record["original_path"],
                        source_class=record["source_class"],
                        true_label=record["true_label"],
                        captured_at=record["captured_at"],
                        capture_timestamp=record["capture_timestamp"],
                        device=record["device"],
                        latitude=record["latitude"],
                        longitude=record["longitude"],
                        altitude=record["altitude"],
                        bearing=record["bearing"],
                        session_id=record["session_id"],
                        plot_id=plot_id,
                        burst_id=burst_id,
                        plant_id=burst_id,
                        group_id=burst_id,
                        burst_position=position,
                        burst_size=len(burst),
                    )
                )

    captures.sort(key=lambda item: (item.capture_timestamp, item.image_path))
    summary = summarize_groups(captures, burst_gap_seconds, plot_radius_metres)
    return captures, summary


def summarize_groups(
    captures: list[CanopyCapture],
    burst_gap_seconds: float,
    plot_radius_metres: float,
) -> dict:
    by_class: dict[str, dict] = {}
    for capture in captures:
        entry = by_class.setdefault(
            capture.true_label,
            {"canopies": 0, "groups": set(), "plots": set(), "sessions": set()},
        )
        entry["canopies"] += 1
        entry["groups"].add(capture.group_id)
        entry["plots"].add(capture.plot_id)
        entry["sessions"].add(capture.session_id)

    group_sizes: dict[str, int] = {}
    for capture in captures:
        group_sizes[capture.group_id] = group_sizes.get(capture.group_id, 0) + 1

    return {
        "burst_gap_seconds": burst_gap_seconds,
        "plot_radius_metres": plot_radius_metres,
        "canopy_count": len(captures),
        "group_count": len(group_sizes),
        "session_count": len({capture.session_id for capture in captures}),
        "plot_count": len({capture.plot_id for capture in captures}),
        "device_count": len({capture.device for capture in captures}),
        "capture_window": {
            "start": captures[0].captured_at if captures else None,
            "end": captures[-1].captured_at if captures else None,
        },
        "by_class": {
            label: {
                "canopies": entry["canopies"],
                "groups": len(entry["groups"]),
                "plots": len(entry["plots"]),
                "sessions": len(entry["sessions"]),
            }
            for label, entry in sorted(by_class.items())
        },
        "group_size_range": [min(group_sizes.values()), max(group_sizes.values())] if group_sizes else [],
    }


def groups_file(output_dir: Path = config.FIELD_SPLIT_DIR) -> Path:
    return output_dir / GROUPS_FILENAME


def write_capture_groups(
    canopy_root: Path = config.CANOPY_DIR,
    output_dir: Path = config.FIELD_SPLIT_DIR,
    burst_gap_seconds: float = config.FIELD_BURST_GAP_SECONDS,
    plot_radius_metres: float = config.FIELD_PLOT_RADIUS_METRES,
) -> tuple[list[CanopyCapture], dict]:
    captures, summary = build_capture_groups(
        canopy_root=canopy_root,
        burst_gap_seconds=burst_gap_seconds,
        plot_radius_metres=plot_radius_metres,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(groups_file(output_dir), "w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "captures": [asdict(capture) for capture in captures]},
            handle,
            indent=2,
        )
    return captures, summary


def load_capture_groups(
    groups_path: Path | None = None,
    canopy_root: Path = config.CANOPY_DIR,
    create_if_missing: bool = True,
) -> list[CanopyCapture]:
    """Load persisted capture groups, deriving them from EXIF if absent."""
    if groups_path is None:
        groups_path = groups_file()
    if not groups_path.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"No capture-group file found at {groups_path}")
        return write_capture_groups(canopy_root=canopy_root, output_dir=groups_path.parent)[0]

    with open(groups_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [CanopyCapture(**record) for record in payload["captures"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover capture metadata and derive canopy grouping keys."
    )
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.FIELD_SPLIT_DIR)
    parser.add_argument("--burst-gap-seconds", type=float, default=config.FIELD_BURST_GAP_SECONDS)
    parser.add_argument("--plot-radius-metres", type=float, default=config.FIELD_PLOT_RADIUS_METRES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captures, summary = write_capture_groups(
        canopy_root=args.canopy_root,
        output_dir=args.output_dir,
        burst_gap_seconds=args.burst_gap_seconds,
        plot_radius_metres=args.plot_radius_metres,
    )
    print(f"Wrote {len(captures)} capture records to {groups_file(args.output_dir)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
