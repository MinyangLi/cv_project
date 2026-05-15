from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.types import CandidateInstance


@dataclass
class Sam3Availability:
    available: bool
    reason: str | None = None


class Sam3Adapter:
    def __init__(self, model_dir: str, device: str) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._imports = None

    def check_availability(self) -> Sam3Availability:
        if not self.model_dir.exists():
            return Sam3Availability(False, f"SAM3 model directory not found: {self.model_dir}")
        try:
            from transformers import (  # type: ignore
                Sam3TrackerVideoModel,
                Sam3TrackerVideoProcessor,
                Sam3VideoModel,
                Sam3VideoProcessor,
            )
            from accelerate import Accelerator  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            return Sam3Availability(
                False,
                "SAM3 code path unavailable. The local model exists, but transformers/accelerate with Sam3* classes "
                f"could not be imported: {exc}",
            )
        self._imports = {
            "Sam3TrackerVideoModel": Sam3TrackerVideoModel,
            "Sam3TrackerVideoProcessor": Sam3TrackerVideoProcessor,
            "Sam3VideoModel": Sam3VideoModel,
            "Sam3VideoProcessor": Sam3VideoProcessor,
            "Accelerator": Accelerator,
        }
        return Sam3Availability(True, None)

    def run_tracker_video_with_box(
        self,
        frames_bgr: list[np.ndarray],
        *,
        prompt_box: tuple[int, int, int, int],
        max_frames: int | None = None,
    ) -> list[np.ndarray]:
        availability = self.check_availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        torch = __import__("torch")
        accelerator = self._imports["Accelerator"]()
        processor = self._imports["Sam3TrackerVideoProcessor"].from_pretrained(str(self.model_dir))
        model = self._imports["Sam3TrackerVideoModel"].from_pretrained(str(self.model_dir)).to(
            accelerator.device, dtype=torch.bfloat16
        )
        masks: list[np.ndarray] = []
        frame_limit = len(frames_bgr) if max_frames is None else min(len(frames_bgr), max_frames)
        inference_session = processor.init_video_session(
            inference_device=accelerator.device,
            dtype=torch.bfloat16,
        )
        for frame_idx in range(frame_limit):
            rgb_frame = cv2.cvtColor(frames_bgr[frame_idx], cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb_frame, device=accelerator.device, return_tensors="pt")
            if frame_idx == 0:
                input_boxes = [[list(prompt_box)]]
                processor.add_inputs_to_inference_session(
                    inference_session=inference_session,
                    frame_idx=0,
                    obj_ids=1,
                    input_boxes=input_boxes,
                    original_size=tuple(inputs.original_sizes[0].tolist()),
                )
            output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
            processed = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=False,
            )[0]
            frame_mask = processed[0, 0].float().detach().cpu().numpy()
            masks.append((frame_mask > 0).astype(np.uint8) * 255)
        return self._pad_masks(masks, frames_bgr, frame_limit)

    def run_text_video_segmentation(
        self,
        frames_bgr: list[np.ndarray],
        *,
        target_text: str,
        max_frames: int | None = None,
    ) -> tuple[list[np.ndarray], list[CandidateInstance] | None]:
        availability = self.check_availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        torch = __import__("torch")
        accelerator = self._imports["Accelerator"]()
        processor = self._imports["Sam3VideoProcessor"].from_pretrained(str(self.model_dir))
        model = self._imports["Sam3VideoModel"].from_pretrained(str(self.model_dir)).to(
            accelerator.device, dtype=torch.bfloat16
        )

        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        inference_session = processor.init_video_session(
            video=rgb_frames,
            inference_device=accelerator.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )
        inference_session = processor.add_text_prompt(inference_session=inference_session, text=target_text)

        masks: list[np.ndarray] = []
        first_frame_candidates: list[CandidateInstance] | None = None
        frame_limit = len(frames_bgr) if max_frames is None else min(len(frames_bgr), max_frames)
        for output in model.propagate_in_video_iterator(inference_session=inference_session, max_frame_num_to_track=frame_limit - 1):
            processed = processor.postprocess_outputs(inference_session, output)
            frame_masks = processed["masks"]
            merged = np.zeros(frame_masks.shape[-2:], dtype=np.uint8)
            for mask in frame_masks:
                merged = np.maximum(merged, (mask > 0).astype(np.uint8) * 255)
            masks.append(merged)
            if output.frame_idx == 0 and first_frame_candidates is None:
                first_frame_candidates = self._extract_candidates(processed)
        return self._pad_masks(masks, frames_bgr, frame_limit), first_frame_candidates

    def _extract_candidates(self, processed: dict[str, Any]) -> list[CandidateInstance]:
        boxes = processed.get("boxes")
        scores = processed.get("scores")
        masks = processed.get("masks")
        object_ids = processed.get("object_ids")
        out: list[CandidateInstance] = []
        if boxes is None or scores is None or masks is None:
            return out
        for idx in range(len(boxes)):
            box = tuple(int(v) for v in boxes[idx].tolist())
            score = float(scores[idx].item() if hasattr(scores[idx], "item") else scores[idx])
            track_id = None
            if object_ids is not None:
                raw = object_ids[idx]
                track_id = int(raw.item() if hasattr(raw, "item") else raw)
            mask = masks[idx]
            if hasattr(mask, "detach"):
                mask = mask.float().detach().cpu().numpy()
            out.append(
                CandidateInstance(
                    bbox=box,
                    score=score,
                    track_id=track_id,
                    mask=(mask > 0).astype(np.uint8) * 255,
                )
            )
        return out

    def _pad_masks(
        self,
        masks: list[np.ndarray],
        frames_bgr: list[np.ndarray],
        frame_limit: int,
    ) -> list[np.ndarray]:
        if not masks:
            h, w = frames_bgr[0].shape[:2]
            return [np.zeros((h, w), dtype=np.uint8) for _ in range(frame_limit)]
        h, w = frames_bgr[0].shape[:2]
        while len(masks) < frame_limit:
            masks.append(np.zeros((h, w), dtype=np.uint8))
        return masks[:frame_limit]
