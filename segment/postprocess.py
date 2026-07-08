"""Post-processing for semantic leaf masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

import config


@dataclass(frozen=True)
class Component:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int


def threshold_mask(probability_mask: np.ndarray, threshold: float = config.SEGMENTATION_THRESHOLD) -> np.ndarray:
    """Convert a probability mask into a binary mask."""
    mask = np.squeeze(probability_mask) >= threshold
    return mask.astype(bool)


def binary_filter(mask: np.ndarray, filter_cls, iterations: int) -> np.ndarray:
    """Apply a PIL binary min/max filter without adding image-processing deps."""
    if iterations <= 0:
        return mask
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(iterations):
        image = image.filter(filter_cls(3))
    return np.asarray(image) > 127


def split_touching_components(mask: np.ndarray) -> np.ndarray:
    """Erode thin bridges before component extraction, then dilate seeds back."""
    eroded = binary_filter(mask, ImageFilter.MinFilter, config.COMPONENT_EROSION_ITERATIONS)
    if not eroded.any():
        return mask
    return binary_filter(eroded, ImageFilter.MaxFilter, config.COMPONENT_DILATION_ITERATIONS)


def connected_components(binary_mask: np.ndarray) -> list[Component]:
    """Find connected components without external image-processing dependencies."""
    visited = np.zeros(binary_mask.shape, dtype=bool)
    height, width = binary_mask.shape
    components: list[Component] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary_mask[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            pixels = []

            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or nx < 0 or ny >= height or nx >= width:
                            continue
                        if visited[ny, nx] or not binary_mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            comp_mask = np.zeros(binary_mask.shape, dtype=bool)
            comp_mask[ys, xs] = True
            components.append(
                Component(
                    mask=comp_mask,
                    bbox=(min(xs), min(ys), max(xs) + 1, max(ys) + 1),
                    area=len(pixels),
                )
            )

    return components


def filter_components(
    components: list[Component],
    image_shape: tuple[int, int],
    min_area_ratio: float = config.MIN_LEAF_AREA_RATIO,
    max_area_ratio: float = config.MAX_LEAF_AREA_RATIO,
) -> list[Component]:
    """Remove tiny noise and implausibly large whole-canopy masks."""
    total_area = image_shape[0] * image_shape[1]
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio
    filtered = [comp for comp in components if min_area <= comp.area <= max_area]
    filtered.sort(key=lambda comp: comp.area, reverse=True)
    return filtered[: config.MAX_LEAVES_PER_IMAGE]


def feather_mask(mask: np.ndarray) -> Image.Image:
    """Create a soft PIL mask for cleaner crop compositing."""
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if config.MASK_FEATHER_RADIUS > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(config.MASK_FEATHER_RADIUS))
    return mask_img


def components_from_probability_mask(probability_mask: np.ndarray) -> list[Component]:
    binary = threshold_mask(probability_mask)
    binary = split_touching_components(binary)
    components = connected_components(binary)
    return filter_components(components, binary.shape)
