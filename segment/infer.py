"""Canopy-to-leaf segmentation inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageOps

import config
import segment.postprocess as postprocess

SEGMENTATION_BACKGROUND = (0, 0, 0)


@dataclass
class LeafInstance:
    crop: Image.Image
    mask: Image.Image
    bbox: tuple[int, int, int, int]
    area: int
    segmentation_bbox: tuple[int, int, int, int]
    segmentation_area: int
    crop_size: tuple[int, int]
    mask_coverage: float
    upsample_factor: float


@dataclass
class CropRejection:
    reason: str
    bbox: tuple[int, int, int, int] | None
    segmentation_bbox: tuple[int, int, int, int]
    segmentation_area: int
    crop_size: tuple[int, int]
    mask_coverage: float
    upsample_factor: float


@dataclass(frozen=True)
class SegmentationTransform:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int]
    scale: float

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        pad_x, pad_y = self.padding
        width, height = self.resized_size
        return pad_x, pad_y, pad_x + width, pad_y + height


def load_segmenter(model_path: Path | None = None) -> keras.Model:
    if model_path is None:
        model_path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME
    return keras.models.load_model(model_path, compile=False)


def fit_segmentation_square(image: Image.Image) -> tuple[Image.Image, SegmentationTransform]:
    """Match segment/data.py preprocessing: preserve aspect ratio and pad."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    resized = image.copy()
    resized.thumbnail(
        (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new(
        "RGB",
        (config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE),
        SEGMENTATION_BACKGROUND,
    )
    padding = (
        (config.SEGMENTATION_IMG_SIZE - resized.width) // 2,
        (config.SEGMENTATION_IMG_SIZE - resized.height) // 2,
    )
    canvas.paste(resized, padding)
    scale = resized.width / original_size[0] if original_size[0] else 1.0
    transform = SegmentationTransform(
        original_size=original_size,
        resized_size=resized.size,
        padding=padding,
        scale=scale,
    )
    return canvas, transform


def prepare_image(image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    resized, _ = fit_segmentation_square(image)
    array = np.asarray(resized).astype(np.float32) / 255.0
    return np.expand_dims(array, axis=0), original_size


def clamp_box(box: tuple[int, int, int, int], bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    left = max(box[0], bounds[0])
    top = max(box[1], bounds[1])
    right = min(box[2], bounds[2])
    bottom = min(box[3], bounds[3])
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def segmentation_box_to_original(
    box: tuple[int, int, int, int],
    transform: SegmentationTransform,
) -> tuple[int, int, int, int] | None:
    """Map a segmentation-canvas bbox back to original image coordinates."""
    content_box = transform.content_box
    clipped = clamp_box(box, content_box)
    if clipped is None:
        return None

    pad_x, pad_y = transform.padding
    original_width, original_height = transform.original_size
    left = math.floor((clipped[0] - pad_x) / transform.scale)
    top = math.floor((clipped[1] - pad_y) / transform.scale)
    right = math.ceil((clipped[2] - pad_x) / transform.scale)
    bottom = math.ceil((clipped[3] - pad_y) / transform.scale)
    return (
        max(0, min(left, original_width - 1)),
        max(0, min(top, original_height - 1)),
        max(1, min(right, original_width)),
        max(1, min(bottom, original_height)),
    )


def crop_stats(
    bbox: tuple[int, int, int, int] | None,
    mask_coverage: float = 0.0,
) -> tuple[tuple[int, int], float]:
    if bbox is None:
        return (0, 0), 0.0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    upsample_factor = config.IMG_SIZE / max(width, height) if max(width, height) > 0 else 0.0
    return (width, height), float(upsample_factor)


def crop_reject_reason(crop_size: tuple[int, int], mask_coverage: float, upsample_factor: float) -> str | None:
    width, height = crop_size
    if width <= 0 or height <= 0:
        return "outside_original_image"
    if min(width, height) < config.MIN_ORIGINAL_CROP_SIZE:
        return "too_small_dimension"
    if width * height < config.MIN_ORIGINAL_CROP_AREA:
        return "too_small_area"
    if max(width, height) / max(1, min(width, height)) > config.MAX_CROP_ASPECT_RATIO:
        return "extreme_aspect_ratio"
    if upsample_factor > config.MAX_CROP_UPSAMPLE_FACTOR:
        return "excessive_upsample"
    if mask_coverage < config.MIN_CROP_MASK_COVERAGE:
        return "low_mask_coverage"
    return None


def crop_component(
    original_image: Image.Image,
    component: postprocess.Component,
    transform: SegmentationTransform,
) -> LeafInstance | CropRejection:
    segmentation_bbox = component.bbox
    original_bbox = segmentation_box_to_original(segmentation_bbox, transform)
    crop_size, upsample_factor = crop_stats(original_bbox)
    if original_bbox is None:
        return CropRejection(
            reason="outside_original_image",
            bbox=None,
            segmentation_bbox=segmentation_bbox,
            segmentation_area=component.area,
            crop_size=crop_size,
            mask_coverage=0.0,
            upsample_factor=upsample_factor,
        )

    content_box = transform.content_box
    clipped_segmentation_bbox = clamp_box(segmentation_bbox, content_box)
    if clipped_segmentation_bbox is None:
        return CropRejection(
            reason="outside_original_image",
            bbox=original_bbox,
            segmentation_bbox=segmentation_bbox,
            segmentation_area=component.area,
            crop_size=crop_size,
            mask_coverage=0.0,
            upsample_factor=upsample_factor,
        )

    mask_crop_array = component.mask[
        clipped_segmentation_bbox[1] : clipped_segmentation_bbox[3],
        clipped_segmentation_bbox[0] : clipped_segmentation_bbox[2],
    ]
    mask_crop = Image.fromarray((mask_crop_array.astype(np.uint8) * 255), mode="L")
    mask_crop = mask_crop.resize(crop_size, Image.Resampling.NEAREST)
    mask_coverage = float(np.asarray(mask_crop).mean() / 255.0)
    crop_size, upsample_factor = crop_stats(original_bbox, mask_coverage)

    reason = crop_reject_reason(crop_size, mask_coverage, upsample_factor)
    if reason is not None:
        return CropRejection(
            reason=reason,
            bbox=original_bbox,
            segmentation_bbox=segmentation_bbox,
            segmentation_area=component.area,
            crop_size=crop_size,
            mask_coverage=mask_coverage,
            upsample_factor=upsample_factor,
        )

    mask_crop = postprocess.feather_mask(np.asarray(mask_crop) > 127)
    crop = original_image.crop(original_bbox)
    background = Image.new("RGB", crop.size, config.NEUTRAL_BACKGROUND_RGB)
    crop = Image.composite(crop, background, mask_crop)
    area = int(np.asarray(mask_crop).sum() / 255)
    return LeafInstance(
        crop=crop,
        mask=mask_crop,
        bbox=original_bbox,
        area=area,
        segmentation_bbox=segmentation_bbox,
        segmentation_area=component.area,
        crop_size=crop_size,
        mask_coverage=mask_coverage,
        upsample_factor=upsample_factor,
    )


def prepare_leaf_for_classifier(crop: Image.Image, image_size: int = config.IMG_SIZE) -> Image.Image:
    """Resize without distorting aspect ratio, padding with the neutral crop background."""
    crop = crop.convert("RGB")
    resized = crop.copy()
    resized.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), config.NEUTRAL_BACKGROUND_RGB)
    canvas.paste(resized, ((image_size - resized.width) // 2, (image_size - resized.height) // 2))
    return canvas


def segment_image_with_audit(
    image: Image.Image,
    model: keras.Model | None = None,
    model_path: Path | None = None,
) -> tuple[list[LeafInstance], list[CropRejection]]:
    """Segment one canopy image, returning accepted crops and rejected components."""
    if model is None:
        model = load_segmenter(model_path)

    original = ImageOps.exif_transpose(image).convert("RGB")
    segmentation_canvas, transform = fit_segmentation_square(original)
    array = np.asarray(segmentation_canvas).astype(np.float32) / 255.0
    probability_mask = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    components = postprocess.components_from_probability_mask(probability_mask)

    instances: list[LeafInstance] = []
    rejections: list[CropRejection] = []
    for component in components:
        result = crop_component(original, component, transform)
        if isinstance(result, CropRejection):
            rejections.append(result)
        else:
            instances.append(result)
    return instances, rejections


def segment_image(
    image: Image.Image,
    model: keras.Model | None = None,
    model_path: Path | None = None,
) -> list[LeafInstance]:
    """Segment one canopy image into leaf crops."""
    instances, _ = segment_image_with_audit(image, model=model, model_path=model_path)
    return instances


def save_leaf_instances(instances: list[LeafInstance], output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, instance in enumerate(instances):
        path = output_dir / f"{prefix}__leaf_{idx:03d}.jpg"
        prepare_leaf_for_classifier(instance.crop).save(path)
        paths.append(path)
    return paths


def segment_file(image_path: Path, output_dir: Path, model_path: Path | None = None) -> list[Path]:
    model = load_segmenter(model_path)
    with Image.open(image_path) as image:
        instances = segment_image(image, model=model)
    return save_leaf_instances(instances, output_dir, image_path.stem)
