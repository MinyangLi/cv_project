from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_YOLO_CLASSES = (
    "person",
    "car",
    "bus",
    "bicycle",
    "motorcycle",
    "truck",
)

TARGET_CLASS_ALIASES = {
    "motorbike": "motorcycle",
    "bike": "bicycle",
}


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    enable_optical_flow: bool = False
    allow_yolo_only_fallback: bool = False
    max_frames: int | None = None
    save_frames: bool = False
    save_debug: bool = False


@dataclass(frozen=True)
class ModelConfig:
    sam3_model_dir: str
    yolo_detector: str = "yolov8n.pt"
    yolo_segmenter: str = "yolov8n-seg.pt"


@dataclass(frozen=True)
class ThresholdConfig:
    det_conf: float = 0.35
    min_box_area_ratio: float = 0.0005
    max_box_area_ratio: float = 0.8
    fallback_min_area: int = 100
    fallback_iou_threshold: float = 0.2
    fallback_center_distance_ratio: float = 0.25
    mask_threshold: float = 0.5
    area_ratio_spike: float = 4.0


@dataclass(frozen=True)
class PostprocessConfig:
    binary_threshold: float = 0.5
    dilate_kernel: int = 7
    dilate_iterations: int = 1
    close_kernel: int = 7
    min_component_area: int = 64


@dataclass(frozen=True)
class PromptConfig:
    target_text: str
    yolo_class: str | None = None
    prompt_box: tuple[int, int, int, int] | None = None
    all_instances: bool = False


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    models: ModelConfig
    thresholds: ThresholdConfig
    postprocess: PostprocessConfig
    prompt: PromptConfig


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    num_frames: int


@dataclass
class CandidateInstance:
    bbox: tuple[int, int, int, int]
    score: float
    track_id: int | None = None
    mask: np.ndarray | None = None
    class_name: str | None = None


@dataclass
class FrameMaskRecord:
    frame_index: int
    mask: np.ndarray
    area: int
    source: str
    bbox: tuple[int, int, int, int] | None = None
    warning: str | None = None


@dataclass
class PipelineWarnings:
    items: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        if message not in self.items:
            self.items.append(message)


@dataclass
class PipelineMetadata:
    input_path: str
    output_mask_video: str
    output_overlay_video: str
    masks_dir: str
    target_text: str
    yolo_class: str | None
    prompt_box: tuple[int, int, int, int] | None
    all_instances: bool
    fps: float
    width: int
    height: int
    num_frames: int
    processed_frames: int = 0
    sam3_available: bool = False
    sam3_mode: str = "uninitialized"
    fallback_frames: list[int] = field(default_factory=list)
    fallback_failed_frames: list[int] = field(default_factory=list)
    frame_mask_areas: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_mask_video": self.output_mask_video,
            "output_overlay_video": self.output_overlay_video,
            "masks_dir": self.masks_dir,
            "target_text": self.target_text,
            "yolo_class": self.yolo_class,
            "prompt_box": list(self.prompt_box) if self.prompt_box is not None else None,
            "all_instances": self.all_instances,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "processed_frames": self.processed_frames,
            "sam3_available": self.sam3_available,
            "sam3_mode": self.sam3_mode,
            "fallback_frames": self.fallback_frames,
            "fallback_failed_frames": self.fallback_failed_frames,
            "frame_mask_areas": self.frame_mask_areas,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class OutputPaths:
    output_mask_video: Path
    output_overlay_video: Path
    output_root: Path
    masks_dir: Path
    frames_dir: Path | None = None

