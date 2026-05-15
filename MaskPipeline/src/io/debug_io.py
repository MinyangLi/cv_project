from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.common.types import PipelineMetadata
from src.io.video_io import save_mask_png


def render_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    frame_index: int,
    source: str,
    bbox: tuple[int, int, int, int] | None = None,
    warning: str | None = None,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    colored = np.zeros_like(frame_bgr)
    colored[:, :, 1] = (mask > 0).astype(np.uint8) * 255
    overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0.0)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    text_lines = [f"frame={frame_index}", f"source={source}", f"area={int(np.count_nonzero(mask))}"]
    if warning:
        text_lines.append(f"warn={warning}")
    for idx, line in enumerate(text_lines):
        cv2.putText(
            overlay,
            line,
            (20, 40 + 28 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def save_debug_frame(path: str | Path, frame_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise IOError(f"Failed to encode debug frame: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def save_mask_frame(path: str | Path, mask: np.ndarray) -> None:
    save_mask_png(path, mask)


def save_metadata(path: str | Path, metadata: PipelineMetadata) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_json(), handle, indent=2, ensure_ascii=False)
