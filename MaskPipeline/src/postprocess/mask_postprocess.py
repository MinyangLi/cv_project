from __future__ import annotations

import cv2
import numpy as np

from src.common.types import PostprocessConfig


def _ensure_binary(mask: np.ndarray, threshold: float) -> np.ndarray:
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.dtype == np.uint8 and set(np.unique(mask)).issubset({0, 255}):
        return mask.copy()
    return (mask.astype(np.float32) > threshold).astype(np.uint8) * 255


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 255
    return out


def postprocess_mask(mask: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    out = _ensure_binary(mask, config.binary_threshold)

    if config.dilate_kernel > 1 and config.dilate_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.dilate_kernel, config.dilate_kernel)
        )
        out = cv2.dilate(out, kernel, iterations=config.dilate_iterations)

    if config.close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

    out = _remove_small_components(out, config.min_component_area)
    return out.astype(np.uint8)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
