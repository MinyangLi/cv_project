from __future__ import annotations

import cv2
import numpy as np

from src.common.types import CandidateInstance, ThresholdConfig
from src.detection.yolo_prompt_detector import bbox_center_distance, bbox_iou
from src.postprocess.mask_postprocess import mask_to_bbox


class YoloSegFallback:
    def __init__(self, model_name: str, device: str, thresholds: ThresholdConfig) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ultralytics is required for YOLO segmentation fallback.") from exc
        self._model = YOLO(model_name)
        self._device = device
        self._thresholds = thresholds

    def segment_candidates(
        self,
        frame_bgr: np.ndarray,
        *,
        yolo_class: str | None,
    ) -> list[CandidateInstance]:
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
        if res.boxes is None or res.masks is None or len(res.boxes) == 0:
            return []

        out: list[CandidateInstance] = []
        h, w = frame_bgr.shape[:2]
        for index in range(len(res.boxes)):
            class_id = int(res.boxes.cls[index].item())
            class_name = str(res.names.get(class_id, str(class_id))).lower()
            if yolo_class is not None and class_name != yolo_class:
                continue
            score = float(res.boxes.conf[index].item())
            xyxy = res.boxes.xyxy[index].detach().cpu().numpy().astype(np.int32)
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            mask = res.masks.data[index].detach().cpu().numpy()
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8) * 255
            track_id = None
            if res.boxes.id is not None:
                track_id = int(res.boxes.id[index].item())
            out.append(
                CandidateInstance(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    track_id=track_id,
                    mask=mask,
                    class_name=class_name,
                )
            )
        return out

    def select_single_instance(
        self,
        candidates: list[CandidateInstance],
        *,
        previous_track_id: int | None,
        previous_bbox: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, int | None, str | None]:
        if not candidates:
            return None, None, None, "fallback_failed:no_candidates"

        if previous_track_id is not None:
            for cand in candidates:
                if cand.track_id == previous_track_id and cand.mask is not None:
                    return cand.mask, cand.bbox, cand.track_id, None

        if previous_bbox is not None:
            by_iou = sorted(candidates, key=lambda c: bbox_iou(previous_bbox, c.bbox), reverse=True)
            best_iou = bbox_iou(previous_bbox, by_iou[0].bbox)
            if best_iou >= self._thresholds.fallback_iou_threshold and by_iou[0].mask is not None:
                cand = by_iou[0]
                return cand.mask, cand.bbox, cand.track_id, None

            h, w = frame_shape[:2]
            max_dist = self._thresholds.fallback_center_distance_ratio * max(h, w)
            by_dist = sorted(candidates, key=lambda c: bbox_center_distance(previous_bbox, c.bbox))
            if bbox_center_distance(previous_bbox, by_dist[0].bbox) <= max_dist and by_dist[0].mask is not None:
                cand = by_dist[0]
                return cand.mask, cand.bbox, cand.track_id, None

        best = max(candidates, key=lambda c: c.score)
        if best.mask is None:
            return None, None, None, "fallback_failed:missing_mask"
        if previous_bbox is None and previous_track_id is None:
            return best.mask, best.bbox, best.track_id, None
        return None, None, None, "fallback_failed:unreliable_match"

    def _class_id_from_name(self, class_name: str) -> int:
        name_to_id = {str(v).lower(): int(k) for k, v in self._model.names.items()}
        if class_name not in name_to_id:
            raise ValueError(f"Class '{class_name}' not found in YOLO model names.")
        return name_to_id[class_name]


def mask_area(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def area_is_suspicious(current_area: int, previous_area: int, area_ratio_spike: float) -> bool:
    if current_area <= 0:
        return True
    if previous_area <= 0:
        return False
    ratio = max(current_area, previous_area) / max(1.0, min(current_area, previous_area))
    return ratio >= area_ratio_spike
