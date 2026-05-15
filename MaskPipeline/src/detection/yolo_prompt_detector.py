from __future__ import annotations

import math

import cv2
import numpy as np

from src.common.types import CandidateInstance, SUPPORTED_YOLO_CLASSES, TARGET_CLASS_ALIASES, ThresholdConfig


def normalize_yolo_class(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = TARGET_CLASS_ALIASES.get(value.strip().lower(), value.strip().lower())
    if normalized not in SUPPORTED_YOLO_CLASSES:
        raise ValueError(
            f"Unsupported yolo class '{value}'. Supported values: {', '.join(SUPPORTED_YOLO_CLASSES)}"
        )
    return normalized


class YoloPromptDetector:
    def __init__(self, model_name: str, device: str, thresholds: ThresholdConfig) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ultralytics is required for YOLO detection fallback.") from exc
        self._model = YOLO(model_name)
        self._device = device
        self._thresholds = thresholds

    def detect(self, frame_bgr: np.ndarray, *, yolo_class: str | None) -> list[CandidateInstance]:
        classes = None
        if yolo_class is not None:
            classes = [self._class_id_from_name(yolo_class)]
        results = self._model.track(
            source=frame_bgr,
            conf=self._thresholds.det_conf,
            device=self._device,
            classes=classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        h, w = frame_bgr.shape[:2]
        out: list[CandidateInstance] = []
        for index in range(len(res.boxes)):
            score = float(res.boxes.conf[index].item())
            class_id = int(res.boxes.cls[index].item())
            class_name = str(res.names.get(class_id, str(class_id))).lower()
            if yolo_class is not None and class_name != yolo_class:
                continue
            xyxy = res.boxes.xyxy[index].detach().cpu().numpy().astype(np.int32)
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1.0, float(h * w))
            if area_ratio < self._thresholds.min_box_area_ratio or area_ratio > self._thresholds.max_box_area_ratio:
                continue
            track_id = None
            if res.boxes.id is not None:
                track_id = int(res.boxes.id[index].item())
            out.append(
                CandidateInstance(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    track_id=track_id,
                    class_name=class_name,
                )
            )
        return out

    def _class_id_from_name(self, class_name: str) -> int:
        name_to_id = {str(v).lower(): int(k) for k, v in self._model.names.items()}
        if class_name not in name_to_id:
            raise ValueError(f"Class '{class_name}' not found in YOLO model names.")
        return name_to_id[class_name]


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / max(union, 1e-6)


def bbox_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    acx = 0.5 * (a[0] + a[2])
    acy = 0.5 * (a[1] + a[3])
    bcx = 0.5 * (b[0] + b[2])
    bcy = 0.5 * (b[1] + b[3])
    return math.hypot(acx - bcx, acy - bcy)


def crop_box_to_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask
