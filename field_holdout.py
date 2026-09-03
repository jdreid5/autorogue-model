"""Construct a genuinely untouched field test set (recommended experiment 1).

The existing `canopy_splits.json` is contaminated three independent ways:

1. Splits are per image with `random.Random(42)`, so near-duplicate frames of one
   plant land on both sides of the train/test boundary.
2. 28 of the 54 LabelMe-annotated canopies are validation/test canopies, and they
   were mixed into segmenter training at 3x sample weight.
3. `select_field_failures.py` added +10 to a canopy's score for being in
   validation/test and scored using model predictions on held-out data.

This module fixes 1 by assigning splits per capture group, and makes 2 and 3
enforceable by emitting a split of record that every fitting stage consults.

Two design choices are worth stating explicitly, because the obvious alternatives
bias the result:

*   Prior contamination does not disqualify a group from the test set. Excluding
    annotated canopies would look safer but would systematically remove the
    hardest canopies, since annotation targeted pipeline failures, leaving an
    optimistically easy test set. Instead every group is eligible, assignment is
    pure grouped randomization, and the audit lists which existing artifacts are
    invalidated and must be regenerated.
*   Selection exposure is measured rather than avoided. No healthy burst is free
    of the 100 already-scored canopies, so avoidance is impossible; but because
    assignment ignores selection score, the audit can verify the test set's
    exposure rate matches the population rate.

The binding limitation this surfaces is statistical, not procedural: grouping
collapses 209 images into far fewer independent plants, so the honest effective
sample size is the group count, not the canopy count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import config
from field_groups import CanopyCapture, groups_file, load_capture_groups
from field_splits import (
    ADAPT_SPLIT,
    QUARANTINE_SPLIT,
    TEST_SPLIT,
    VALIDATION_SPLIT,
)

MANIFEST_FILENAME = "untouched_field_test_manifest.json"
FOLDS_FILENAME = "untouched_canopy_folds.json"

# Touches that mean a model was fitted on the canopy. Any of these appearing in
# the test set invalidates the corresponding artifact until it is regenerated.
FITTING_TOUCHES = ("segmentation_annotation", "segmenter_training", "classifier_adaptation")
# Calibration fits decision thresholds rather than weights, but still leaks.
TUNING_TOUCHES = ("threshold_calibration",)
# Selection never fed a fit; it only chose what to annotate.
EXPOSURE_TOUCHES = ("failure_selection",)

STALE_ARTIFACTS = {
    "segmentation_annotation": "models/autorogue_leaf_segmenter.keras",
    "segmenter_training": "models/autorogue_leaf_segmenter.keras",
    "classifier_adaptation": "models/autorogue_leaf_classifier_field_adapted.keras",
    "threshold_calibration": "outputs/field_validation_results*.json thresholds",
    "failure_selection": "outputs/field_segmentation_annotation_candidates.json",
}


def file_fingerprint(path: Path) -> dict:
    """Hash and stat a file so a result can be tied to exact inputs."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "sha256": digest,
        "sha256_12": digest[:12],
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "bytes": stat.st_size,
    }


def annotation_stem(annotation_path: Path) -> str:
    """Recover the canopy stem from a `{source_class}__{stem}` annotation name."""
    name = annotation_path.stem
    return name.split("__", 1)[1] if "__" in name else name


def collect_touch_ledger(
    field_root: Path = config.FIELD_LEAF_DIR,
    annotation_dir: Path = config.FIELD_SEGMENTATION_DATA_DIR / "raw",
    segmenter_image_dir: Path = config.FIELD_SEGMENTATION_IMAGE_DIR,
    legacy_split_path: Path | None = None,
    candidates_path: Path | None = None,
    validation_results_paths: list[Path] | None = None,
) -> tuple[dict[str, set[str]], dict]:
    """Record which canopies each prior process consumed, keyed by filename stem."""
    if legacy_split_path is None:
        legacy_split_path = config.FIELD_SPLIT_DIR / "canopy_splits.json"
    if candidates_path is None:
        candidates_path = config.OUTPUTS_DIR / "field_segmentation_annotation_candidates.json"
    if validation_results_paths is None:
        validation_results_paths = sorted(config.OUTPUTS_DIR.glob("field_validation_results*.json"))

    ledger: dict[str, set[str]] = {name: set() for name in
                                   FITTING_TOUCHES + TUNING_TOUCHES + EXPOSURE_TOUCHES}
    sources: dict[str, list[str]] = defaultdict(list)

    if annotation_dir.exists():
        for path in sorted(annotation_dir.glob("*.json")):
            if path.name == "annotation_manifest.json":
                continue
            ledger["segmentation_annotation"].add(annotation_stem(path))
        sources["segmentation_annotation"].append(str(annotation_dir))

    if segmenter_image_dir.exists():
        for path in sorted(segmenter_image_dir.iterdir()):
            if path.is_file():
                ledger["segmenter_training"].add(annotation_stem(path))
        sources["segmenter_training"].append(str(segmenter_image_dir))

    manifest_path = field_root / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("split") == ADAPT_SPLIT:
                    ledger["classifier_adaptation"].add(Path(record["canopy_path"]).stem)
        sources["classifier_adaptation"].append(str(manifest_path))

    if Path(legacy_split_path).exists():
        with open(legacy_split_path, "r", encoding="utf-8") as handle:
            for record in json.load(handle):
                if record.get("split") == VALIDATION_SPLIT:
                    ledger["threshold_calibration"].add(Path(record["image_path"]).stem)
        sources["threshold_calibration"].append(str(legacy_split_path))

    for path in validation_results_paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get(VALIDATION_SPLIT, {}).get("predictions", []):
            ledger["threshold_calibration"].add(Path(record["image_path"]).stem)
        sources["threshold_calibration"].append(str(path))

    if Path(candidates_path).exists():
        with open(candidates_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get("candidates", []):
            ledger["failure_selection"].add(Path(record["canopy_path"]).stem)
        sources["failure_selection"].append(str(candidates_path))

    provenance = {
        "sources": {key: value for key, value in sorted(sources.items())},
        "counts": {key: len(value) for key, value in sorted(ledger.items())},
    }
    return ledger, provenance


def choose_groups_for_target(
    groups: list[tuple[str, int]],
    target: int,
    rng: random.Random,
) -> list[str]:
    """Pick whole groups whose canopy count lands as close as possible to `target`."""
    shuffled = groups[:]
    rng.shuffle(shuffled)

    chosen: list[str] = []
    total = 0
    for group_id, size in shuffled:
        if total + size <= target:
            chosen.append(group_id)
            total += size

    # Allow a single overshoot when it lands closer to the target than stopping short.
    if total < target:
        remaining = [(gid, size) for gid, size in shuffled if gid not in set(chosen)]
        if remaining:
            best_id, best_size = min(remaining, key=lambda item: abs(total + item[1] - target))
            if abs(total + best_size - target) < abs(total - target):
                chosen.append(best_id)
    return chosen


def assign_group_splits(
    captures: list[CanopyCapture],
    seed: int = config.SEED,
    test_fraction: float = config.FIELD_TEST_FRACTION,
    val_fraction: float = config.FIELD_VAL_FRACTION,
    excluded_from_test: set[str] | None = None,
) -> dict[str, str]:
    """Assign every capture group to a split, stratified by class.

    Assignment depends only on the grouping keys and the seed, never on model
    predictions or prior contamination, so the resulting test set is an unbiased
    sample of the capture groups.
    """
    excluded_from_test = excluded_from_test or set()

    groups_by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for capture in captures:
        groups_by_class[capture.true_label][capture.group_id] += 1

    split_of_group: dict[str, str] = {}
    for label in sorted(groups_by_class):
        sizes = groups_by_class[label]
        class_total = sum(sizes.values())
        rng = random.Random(f"{seed}:{label}")

        eligible = [(gid, size) for gid, size in sorted(sizes.items()) if gid not in excluded_from_test]
        test_groups = set(choose_groups_for_target(eligible, round(class_total * test_fraction), rng))

        remaining = [(gid, size) for gid, size in sorted(sizes.items()) if gid not in test_groups]
        val_groups = set(choose_groups_for_target(remaining, round(class_total * val_fraction), rng))

        for group_id in sizes:
            if group_id in test_groups:
                split_of_group[group_id] = TEST_SPLIT
            elif group_id in val_groups:
                split_of_group[group_id] = VALIDATION_SPLIT
            else:
                split_of_group[group_id] = ADAPT_SPLIT
    return split_of_group


def apply_buffer_quarantine(
    captures: list[CanopyCapture],
    split_of_canopy: dict[str, str],
    buffer_seconds: float,
) -> tuple[dict[str, str], list[dict]]:
    """Quarantine non-test canopies captured within `buffer_seconds` of a test canopy.

    Burst boundaries are a heuristic for plant identity. If the operator paused
    slightly too long mid-plant, one plant becomes two bursts and could straddle
    the split; this buffer absorbs that error.
    """
    if buffer_seconds <= 0:
        return dict(split_of_canopy), []

    by_cohort: dict[tuple[str, str], list[CanopyCapture]] = defaultdict(list)
    for capture in captures:
        by_cohort[(capture.session_id, capture.source_class)].append(capture)

    updated = dict(split_of_canopy)
    quarantined: list[dict] = []
    for cohort in by_cohort.values():
        test_times = [
            capture.capture_timestamp
            for capture in cohort
            if split_of_canopy[capture.image_path] == TEST_SPLIT
        ]
        if not test_times:
            continue
        for capture in cohort:
            if split_of_canopy[capture.image_path] == TEST_SPLIT:
                continue
            gap = min(abs(capture.capture_timestamp - t) for t in test_times)
            if gap <= buffer_seconds:
                updated[capture.image_path] = QUARANTINE_SPLIT
                quarantined.append(
                    {
                        "image_path": capture.image_path,
                        "group_id": capture.group_id,
                        "previous_split": split_of_canopy[capture.image_path],
                        "seconds_from_nearest_test_canopy": round(gap, 3),
                    }
                )
    return updated, quarantined


def assign_cv_folds(
    captures: list[CanopyCapture],
    n_folds: int = config.FIELD_CV_FOLDS,
    seed: int = config.SEED,
) -> dict[str, int]:
    """Assign whole capture groups to group-disjoint, class-balanced CV folds.

    The single holdout leaves only two independent healthy plants in test. Rotating
    every group through the test position uses all seven, which is the difference
    between a metric with a +/-50% interval and one worth acting on.
    """
    fold_of_group: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for capture in captures:
        by_class[capture.true_label][capture.group_id] += 1

    for label in sorted(by_class):
        sizes = by_class[label]
        rng = random.Random(f"{seed}:cv:{label}")
        group_ids = sorted(sizes)
        rng.shuffle(group_ids)
        # Largest first, into whichever fold is currently emptiest for this class.
        group_ids.sort(key=lambda gid: sizes[gid], reverse=True)
        loads = [0] * n_folds
        for group_id in group_ids:
            fold = min(range(n_folds), key=lambda index: (loads[index], index))
            fold_of_group[group_id] = fold
            loads[fold] += sizes[group_id]
    return fold_of_group


def summarize_cv_folds(
    captures: list[CanopyCapture],
    fold_of_group: dict[str, int],
    n_folds: int,
    buffer_seconds: float,
) -> list[dict]:
    """Report, per fold, the test cohort and the training canopies the buffer costs."""
    summaries = []
    for fold in range(n_folds):
        split_of_canopy = {
            capture.image_path: (TEST_SPLIT if fold_of_group[capture.group_id] == fold else ADAPT_SPLIT)
            for capture in captures
        }
        buffered, quarantined = apply_buffer_quarantine(captures, split_of_canopy, buffer_seconds)
        test_captures = [c for c in captures if buffered[c.image_path] == TEST_SPLIT]
        summaries.append(
            {
                "fold": fold,
                "test_canopies": len(test_captures),
                "test_groups": len({c.group_id for c in test_captures}),
                "test_groups_by_class": {
                    label: len({c.group_id for c in test_captures if c.true_label == label})
                    for label in sorted({c.true_label for c in captures})
                },
                "test_canopies_by_class": dict(Counter(c.true_label for c in test_captures)),
                "train_canopies": sum(1 for split in buffered.values() if split == ADAPT_SPLIT),
                "quarantined_canopies": len(quarantined),
            }
        )
    return summaries


def _wilson_half_width(n: int, proportion: float = 0.5, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    denominator = 1 + z**2 / n
    margin = z * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
    return margin / denominator


def audit_split(
    captures: list[CanopyCapture],
    split_of_canopy: dict[str, str],
    ledger: dict[str, set[str]],
    buffer_seconds: float,
) -> tuple[list[dict], dict]:
    """Check the split for leakage and quantify what it can actually measure."""
    by_path = {capture.image_path: capture for capture in captures}
    checks: list[dict] = []

    def add(name: str, status: str, detail: str, **extra) -> None:
        checks.append({"check": name, "status": status, "detail": detail, **extra})

    unassigned = [c.image_path for c in captures if c.image_path not in split_of_canopy]
    add(
        "every_canopy_assigned",
        "pass" if not unassigned else "fail",
        f"{len(captures) - len(unassigned)}/{len(captures)} canopies assigned to a split",
        unassigned=unassigned[:10],
    )

    splits_per_group: dict[str, set[str]] = defaultdict(set)
    for path, split in split_of_canopy.items():
        capture = by_path[path]
        # Quarantine is a per-canopy overlay, so it never counts as a group split.
        if split != QUARANTINE_SPLIT:
            splits_per_group[capture.group_id].add(split)
    straddling = sorted(gid for gid, splits in splits_per_group.items() if len(splits) > 1)
    add(
        "no_capture_group_spans_splits",
        "pass" if not straddling else "fail",
        f"{len(straddling)} of {len(splits_per_group)} capture groups span more than one split",
        groups=straddling[:10],
    )

    test_paths = [p for p, s in split_of_canopy.items() if s == TEST_SPLIT]
    trainable = [p for p, s in split_of_canopy.items() if s == ADAPT_SPLIT]
    calibration = [p for p, s in split_of_canopy.items() if s == VALIDATION_SPLIT]

    min_gap = None
    for test_path in test_paths:
        test_capture = by_path[test_path]
        for other_path in trainable + calibration:
            other = by_path[other_path]
            if other.session_id != test_capture.session_id or other.source_class != test_capture.source_class:
                continue
            gap = abs(other.capture_timestamp - test_capture.capture_timestamp)
            min_gap = gap if min_gap is None else min(min_gap, gap)
    add(
        "temporal_buffer_respected",
        "pass" if min_gap is None or min_gap > buffer_seconds else "fail",
        f"nearest non-test canopy is {min_gap:.1f}s from a test canopy "
        f"(buffer {buffer_seconds:.0f}s)" if min_gap is not None
        else "no same-cohort non-test canopy to compare against",
        min_seconds_to_non_test=round(min_gap, 3) if min_gap is not None else None,
    )

    test_stems = {Path(p).stem for p in test_paths}
    remediation: list[dict] = []
    for touch in FITTING_TOUCHES + TUNING_TOUCHES:
        hit = sorted(test_stems & ledger.get(touch, set()))
        status = "pass" if not hit else "action_required"
        add(
            f"test_clean_of_{touch}",
            status,
            f"{len(hit)} test canopies were consumed by {touch}; the artifact it "
            f"produced is invalid until regenerated under this split"
            if hit else f"no test canopy was consumed by {touch}",
            canopies=hit[:10],
            stale_artifact=STALE_ARTIFACTS[touch] if hit else None,
        )
        if hit:
            remediation.append(
                {"touch": touch, "test_canopies_affected": len(hit),
                 "stale_artifact": STALE_ARTIFACTS[touch]}
            )

    exposed = ledger.get("failure_selection", set())
    test_rate = len(test_stems & exposed) / len(test_stems) if test_stems else 0.0
    population_rate = len(exposed) / len(captures) if captures else 0.0
    drift = abs(test_rate - population_rate)
    add(
        "selection_exposure_representative",
        "pass" if drift <= 0.15 else "warn",
        f"{test_rate:.1%} of test canopies were scored by failure selection versus "
        f"{population_rate:.1%} of all canopies; assignment ignored selection score, "
        f"so the test set is not enriched for pipeline failures",
        test_rate=round(test_rate, 4),
        population_rate=round(population_rate, 4),
    )

    shared_plots = {
        by_path[p].plot_id for p in test_paths
    } & {by_path[p].plot_id for p in trainable}
    add(
        "test_plots_disjoint_from_training",
        "pass" if not shared_plots else "warn",
        f"{len(shared_plots)} plot(s) contribute to both test and training; all healthy "
        f"canopies share one GPS fix, so plot-level disjointness is unachievable for that class"
        if shared_plots else "test plots do not appear in training",
        shared_plots=sorted(shared_plots),
    )

    per_split_groups = defaultdict(set)
    per_split_class = defaultdict(Counter)
    for path, split in split_of_canopy.items():
        capture = by_path[path]
        per_split_groups[split].add(capture.group_id)
        per_split_class[split][capture.true_label] += 1

    test_groups_by_class = Counter(
        {
            label: len({by_path[p].group_id for p in test_paths if by_path[p].true_label == label})
            for label in sorted({by_path[p].true_label for p in test_paths})
        }
    )
    independent_test_groups = len({by_path[p].group_id for p in test_paths})
    scarcest_class, scarcest_count = min(test_groups_by_class.items(), key=lambda item: item[1])
    add(
        "sufficient_independent_test_groups",
        "pass" if scarcest_count >= 5 else "warn",
        f"the test set holds {independent_test_groups} independent capture groups across "
        f"{len(test_paths)} canopies, but only {scarcest_count} of class {scarcest_class!r}; "
        f"per-class metrics resolve to no better than "
        f"+/-{_wilson_half_width(scarcest_count):.0%} at 95% confidence, so read the "
        f"grouped cross-validation folds instead of this holdout alone",
        independent_groups=independent_test_groups,
        groups_by_class=dict(test_groups_by_class),
    )

    statistics = {
        "canopies_per_split": {split: len([p for p, s in split_of_canopy.items() if s == split])
                               for split in sorted(set(split_of_canopy.values()))},
        "groups_per_split": {split: len(groups) for split, groups in sorted(per_split_groups.items())},
        "class_counts_per_split": {split: dict(counter) for split, counter in sorted(per_split_class.items())},
        "test_groups_by_class": dict(test_groups_by_class),
        "effective_sample_size": {
            "note": "Canopies within a capture group are repeated shots of one plant, "
                    "so the independent unit is the group, not the canopy.",
            "test_canopies": len(test_paths),
            "test_independent_groups": independent_test_groups,
            "accuracy_resolution_95ci": round(_wilson_half_width(independent_test_groups), 4),
        },
        "remediation_required": remediation,
    }
    return checks, statistics


def build_untouched_split(
    canopy_root: Path = config.CANOPY_DIR,
    groups_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    seed: int = config.SEED,
    test_fraction: float = config.FIELD_TEST_FRACTION,
    val_fraction: float = config.FIELD_VAL_FRACTION,
    buffer_seconds: float = config.FIELD_HOLDOUT_BUFFER_SECONDS,
    n_folds: int = config.FIELD_CV_FOLDS,
    exclude_annotated_from_test: bool = False,
    allow_failed_checks: bool = False,
) -> dict:
    """Build, audit, and persist the untouched field test split."""
    if groups_path is None:
        groups_path = groups_file()
    if output_path is None:
        output_path = config.UNTOUCHED_SPLIT_PATH
    if manifest_path is None:
        manifest_path = config.OUTPUTS_DIR / MANIFEST_FILENAME
    folds_path = output_path.parent / FOLDS_FILENAME

    captures = load_capture_groups(groups_path=groups_path, canopy_root=canopy_root)
    ledger, ledger_provenance = collect_touch_ledger()

    excluded_from_test: set[str] = set()
    if exclude_annotated_from_test:
        annotated = ledger["segmentation_annotation"]
        excluded_from_test = {
            capture.group_id for capture in captures if Path(capture.image_path).stem in annotated
        }

    split_of_group = assign_group_splits(
        captures,
        seed=seed,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        excluded_from_test=excluded_from_test,
    )
    split_of_canopy = {capture.image_path: split_of_group[capture.group_id] for capture in captures}
    split_of_canopy, quarantined = apply_buffer_quarantine(captures, split_of_canopy, buffer_seconds)

    checks, statistics = audit_split(captures, split_of_canopy, ledger, buffer_seconds)
    failures = [check for check in checks if check["status"] == "fail"]
    if failures and not allow_failed_checks:
        raise RuntimeError(
            "Untouched split failed leakage checks and was not written: "
            + "; ".join(f"{check['check']}: {check['detail']}" for check in failures)
        )

    records = [
        {
            "image_path": capture.image_path,
            "true_label": capture.true_label,
            "source_class": capture.source_class,
            "split": split_of_canopy[capture.image_path],
            "group_id": capture.group_id,
            "plot_id": capture.plot_id,
            "session_id": capture.session_id,
            "captured_at": capture.captured_at,
        }
        for capture in captures
    ]
    records.sort(key=lambda item: (item["split"], item["true_label"], item["image_path"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    fold_of_group = assign_cv_folds(captures, n_folds=n_folds, seed=seed)
    fold_summaries = summarize_cv_folds(captures, fold_of_group, n_folds, buffer_seconds)
    with open(folds_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_folds": n_folds,
                "buffer_seconds": buffer_seconds,
                "folds": fold_summaries,
                "assignments": [
                    {
                        "image_path": capture.image_path,
                        "true_label": capture.true_label,
                        "source_class": capture.source_class,
                        "group_id": capture.group_id,
                        "fold": fold_of_group[capture.group_id],
                    }
                    for capture in captures
                ],
            },
            handle,
            indent=2,
        )

    touched_by_split: dict[str, dict[str, int]] = {}
    for split in sorted(set(split_of_canopy.values())):
        stems = {Path(p).stem for p, s in split_of_canopy.items() if s == split}
        touched_by_split[split] = {touch: len(stems & names) for touch, names in sorted(ledger.items())}

    manifest = {
        "experiment": "1 · genuinely untouched field test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_file": str(output_path),
        "cv_folds_file": str(folds_path),
        "cv_folds": fold_summaries,
        "parameters": {
            "seed": seed,
            "test_fraction": test_fraction,
            "val_fraction": val_fraction,
            "buffer_seconds": buffer_seconds,
            "n_folds": n_folds,
            "burst_gap_seconds": config.FIELD_BURST_GAP_SECONDS,
            "plot_radius_metres": config.FIELD_PLOT_RADIUS_METRES,
            "exclude_annotated_from_test": exclude_annotated_from_test,
            "grouping_keys": ["session_id", "plot_id", "burst_id", "plant_id"],
        },
        "checks": checks,
        "statistics": statistics,
        "touch_ledger": ledger_provenance,
        "touched_canopies_by_split": touched_by_split,
        "quarantined_canopies": quarantined,
        "provenance": provenance_block(groups_path, output_path),
        "caveats": [
            "All 209 canopies come from one date, one device, one variety, and one "
            "operator run, so this split measures generalization to new plants only, "
            "not to new sessions, sites, varieties, devices, or growth stages.",
            "Plant identity is inferred from capture bursts rather than recorded, so "
            "a plant photographed in two separated bursts could still straddle the "
            "split; the temporal buffer is the mitigation, not a proof.",
            "All healthy canopies share a single GPS fix, so plot-level holdout is "
            "impossible for that class and burst grouping carries the whole guarantee.",
            "Every canopy was scored by select_field_failures.py using model "
            "predictions. Assignment ignores that score, so the test set is not "
            "enriched for failures, but the exposure cannot be undone.",
            "Existing checkpoints and calibrated thresholds predate this split. Any "
            "metric read from them is not leak-free; see statistics.remediation_required.",
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Untouched field split written to {output_path}")
    print(f"Grouped cross-validation folds written to {folds_path}")
    print(f"Audit manifest written to {manifest_path}")
    return manifest


def provenance_block(groups_path: Path, split_path: Path) -> dict:
    """Hash every input that determines the split, plus the models it invalidates."""
    pipeline_models = {
        "segmenter": config.MODELS_DIR / config.SEGMENTER_MODEL_NAME,
        "classifier": config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME,
        "classifier_field_adapted": config.MODELS_DIR / "autorogue_leaf_classifier_field_adapted.keras",
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip())
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        commit, dirty = None, None

    return {
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "split_definition_code": {
            name: file_fingerprint(Path(name))
            for name in ("field_groups.py", "field_holdout.py", "field_splits.py", "config.py")
        },
        "capture_groups_file": file_fingerprint(groups_path),
        "split_file": file_fingerprint(split_path),
        "models": {name: file_fingerprint(path) for name, path in pipeline_models.items()},
        "all_model_checkpoints": [
            file_fingerprint(path) for path in sorted(config.MODELS_DIR.glob("*.keras"))
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leak-free, capture-grouped field test split."
    )
    parser.add_argument("--canopy-root", type=Path, default=config.CANOPY_DIR)
    parser.add_argument("--groups-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=config.UNTOUCHED_SPLIT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=config.OUTPUTS_DIR / MANIFEST_FILENAME)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--test-fraction", type=float, default=config.FIELD_TEST_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=config.FIELD_VAL_FRACTION)
    parser.add_argument("--buffer-seconds", type=float, default=config.FIELD_HOLDOUT_BUFFER_SECONDS)
    parser.add_argument("--n-folds", type=int, default=config.FIELD_CV_FOLDS)
    parser.add_argument(
        "--exclude-annotated-from-test",
        action="store_true",
        help="Keep annotated canopies out of test to preserve the annotation budget. "
             "Biases the test set easy, because annotation targeted pipeline failures.",
    )
    parser.add_argument("--allow-failed-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_untouched_split(
        canopy_root=args.canopy_root,
        groups_path=args.groups_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        seed=args.seed,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        buffer_seconds=args.buffer_seconds,
        n_folds=args.n_folds,
        exclude_annotated_from_test=args.exclude_annotated_from_test,
        allow_failed_checks=args.allow_failed_checks,
    )
    for check in manifest["checks"]:
        print(f"  [{check['status']:>15}] {check['check']}: {check['detail']}")
    print(json.dumps(manifest["statistics"], indent=2))


if __name__ == "__main__":
    main()
