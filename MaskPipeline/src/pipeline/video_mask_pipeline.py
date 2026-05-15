from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.common.types import AppConfig, FrameMaskRecord, OutputPaths, PipelineMetadata, PipelineWarnings
from src.detection.yolo_prompt_detector import YoloPromptDetector, bbox_iou, normalize_yolo_class
from src.io.debug_io import render_overlay, save_debug_frame, save_mask_frame, save_metadata
from src.io.video_io import H264VideoWriter, inspect_video, load_video_frames, resolve_output_paths
from src.postprocess.mask_postprocess import mask_to_bbox, postprocess_mask
from src.refine.optical_flow_refine import refine_with_optical_flow
from src.segmentation.sam3_adapter import Sam3Adapter
from src.segmentation.yolo_seg_fallback import YoloSegFallback, area_is_suspicious, mask_area


@dataclass
class PipelineResult:
    metadata: PipelineMetadata
    output_paths: OutputPaths


def _select_best_candidate(
    sam3_candidates,
    *,
    prompt_box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if not sam3_candidates:
        return prompt_box
    if prompt_box is not None:
        scored = sorted(sam3_candidates, key=lambda c: bbox_iou(prompt_box, c.bbox), reverse=True)
        return scored[0].bbox
    best = max(sam3_candidates, key=lambda c: c.score)
    return best.bbox


def run_video_mask_pipeline(
    input_video: str | Path,
    output_video: str | Path | None,
    config: AppConfig,
) -> PipelineResult:
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_paths = resolve_output_paths(input_path, output_video)
    video_info = inspect_video(input_path)
    frames = load_video_frames(input_path, max_frames=config.runtime.max_frames)
    if not frames:
        raise RuntimeError(f"No frames loaded from {input_path}")

    warnings = PipelineWarnings()
    yolo_class = normalize_yolo_class(config.prompt.yolo_class)
    if config.prompt.prompt_box is None and not config.prompt.all_instances:
        warnings.add("No prompt-box provided; the system will auto-select one instance and it may not match user intent.")

    metadata = PipelineMetadata(
        input_path=str(input_path),
        output_mask_video=str(output_paths.output_mask_video),
        output_overlay_video=str(output_paths.output_overlay_video),
        masks_dir=str(output_paths.masks_dir),
        target_text=config.prompt.target_text,
        yolo_class=yolo_class,
        prompt_box=config.prompt.prompt_box,
        all_instances=config.prompt.all_instances,
        fps=video_info.fps,
        width=video_info.width,
        height=video_info.height,
        num_frames=len(frames),
        warnings=warnings.items.copy(),
    )

    sam3 = Sam3Adapter(config.models.sam3_model_dir, config.runtime.device)
    availability = sam3.check_availability()
    metadata.sam3_available = availability.available

    records: list[FrameMaskRecord] = []
    previous_bbox = config.prompt.prompt_box
    previous_track_id = None
    previous_frame = None
    previous_area = 0

    yolo_fallback = None
    if config.runtime.allow_yolo_only_fallback or not availability.available:
        if not availability.available:
            metadata.errors.append(availability.reason or "SAM3 unavailable")
        try:
            yolo_fallback = YoloSegFallback(config.models.yolo_segmenter, config.runtime.device, config.thresholds)
            metadata.sam3_mode = "yolo_only_fallback" if config.runtime.allow_yolo_only_fallback or not availability.available else "tracker_video"
        except Exception as exc:
            if not availability.available:
                raise RuntimeError(
                    f"SAM3 unavailable and YOLO fallback could not be initialized: {exc}"
                ) from exc
            raise

    if availability.available and not config.runtime.allow_yolo_only_fallback:
        if config.prompt.prompt_box is not None:
            metadata.sam3_mode = "tracker_video"
            raw_masks = sam3.run_tracker_video_with_box(
                frames,
                prompt_box=config.prompt.prompt_box,
                max_frames=config.runtime.max_frames,
            )
        else:
            metadata.sam3_mode = "text_video"
            raw_masks, sam3_candidates = sam3.run_text_video_segmentation(
                frames,
                target_text=config.prompt.target_text,
                max_frames=config.runtime.max_frames,
            )
            if not config.prompt.all_instances:
                selected_box = _select_best_candidate(sam3_candidates, prompt_box=None)
                if selected_box is not None:
                    metadata.sam3_mode = "tracker_video"
                    raw_masks = sam3.run_tracker_video_with_box(
                        frames,
                        prompt_box=selected_box,
                        max_frames=config.runtime.max_frames,
                    )
                    previous_bbox = selected_box
                    warnings.add(
                        "Auto-selected one instance from target-text results; use --prompt-box to lock the intended object."
                    )
                    metadata.warnings = warnings.items.copy()
        if yolo_fallback is None:
            try:
                yolo_fallback = YoloSegFallback(config.models.yolo_segmenter, config.runtime.device, config.thresholds)
            except Exception:
                yolo_fallback = None
    else:
        raw_masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for frame in frames]

    if yolo_fallback is None and (config.runtime.allow_yolo_only_fallback or not availability.available):
        raise RuntimeError(
            "YOLO-only fallback requested or needed, but ultralytics is unavailable. Install required dependencies first."
        )

    mask_writer = H264VideoWriter(
        output_paths.output_mask_video, width=video_info.width, height=video_info.height, fps=video_info.fps
    )
    overlay_writer = H264VideoWriter(
        output_paths.output_overlay_video, width=video_info.width, height=video_info.height, fps=video_info.fps
    )

    try:
        for frame_index, frame in enumerate(frames):
            raw_mask = raw_masks[frame_index] if frame_index < len(raw_masks) else np.zeros(frame.shape[:2], dtype=np.uint8)
            source = metadata.sam3_mode
            warning = None
            current_bbox = mask_to_bbox(raw_mask) or previous_bbox
            current_area = mask_area(raw_mask)

            needs_fallback = (
                current_area < config.thresholds.fallback_min_area
                or area_is_suspicious(current_area, previous_area, config.thresholds.area_ratio_spike)
            )

            if needs_fallback and yolo_fallback is not None:
                candidates = yolo_fallback.segment_candidates(frame, yolo_class=yolo_class)
                selected_mask, selected_bbox, selected_track_id, fallback_warning = yolo_fallback.select_single_instance(
                    candidates,
                    previous_track_id=previous_track_id,
                    previous_bbox=previous_bbox,
                    frame_shape=frame.shape,
                )
                if selected_mask is not None:
                    raw_mask = selected_mask
                    current_bbox = selected_bbox
                    previous_track_id = selected_track_id
                    source = "yolo_fallback"
                    metadata.fallback_frames.append(frame_index)
                else:
                    warning = fallback_warning
                    metadata.fallback_failed_frames.append(frame_index)
            if config.runtime.enable_optical_flow:
                raw_mask = refine_with_optical_flow(previous_frame, frame, raw_mask.astype(np.uint8))
            refined = postprocess_mask(raw_mask, config.postprocess)
            refined_bbox = mask_to_bbox(refined) or current_bbox
            area = int(np.count_nonzero(refined))

            metadata.frame_mask_areas.append(area)
            metadata.processed_frames += 1

            if config.runtime.save_frames:
                save_mask_frame(output_paths.masks_dir / f"{frame_index:05d}.png", refined)
            overlay = render_overlay(
                frame,
                refined,
                frame_index=frame_index,
                source=source,
                bbox=refined_bbox,
                warning=warning,
            )
            if config.runtime.save_debug and output_paths.frames_dir is not None:
                save_debug_frame(output_paths.frames_dir / f"{frame_index:05d}.png", overlay)

            mask_writer.write(cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR))
            overlay_writer.write(overlay)

            records.append(
                FrameMaskRecord(
                    frame_index=frame_index,
                    mask=refined,
                    area=area,
                    source=source,
                    bbox=refined_bbox,
                    warning=warning,
                )
            )
            previous_frame = frame
            previous_bbox = refined_bbox or previous_bbox
            previous_area = area
    finally:
        mask_writer.close()
        overlay_writer.close()

    metadata.warnings = warnings.items.copy()
    save_metadata(output_paths.output_root / "meta.json", metadata)
    return PipelineResult(metadata=metadata, output_paths=output_paths)
