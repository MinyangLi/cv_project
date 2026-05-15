from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.common.types import (
    AppConfig,
    ModelConfig,
    PostprocessConfig,
    PromptConfig,
    RuntimeConfig,
    ThresholdConfig,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mask_video.yaml"


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_app_config(
    raw_config: dict[str, Any],
    *,
    target_text: str,
    yolo_class: str | None,
    prompt_box: tuple[int, int, int, int] | None,
    all_instances: bool,
    device: str | None,
    enable_optical_flow: bool | None,
    allow_yolo_only_fallback: bool,
    max_frames: int | None,
    save_frames: bool,
    save_debug: bool,
) -> AppConfig:
    runtime_cfg = raw_config.get("runtime", {})
    models_cfg = raw_config.get("models", {})
    thresholds_cfg = raw_config.get("thresholds", {})
    post_cfg = raw_config.get("postprocess", {})

    return AppConfig(
        runtime=RuntimeConfig(
            device=device or str(runtime_cfg.get("device", "cpu")),
            enable_optical_flow=(
                bool(enable_optical_flow)
                if enable_optical_flow is not None
                else bool(runtime_cfg.get("enable_optical_flow", False))
            ),
            allow_yolo_only_fallback=allow_yolo_only_fallback or bool(
                runtime_cfg.get("allow_yolo_only_fallback", False)
            ),
            max_frames=max_frames if max_frames is not None else runtime_cfg.get("max_frames"),
            save_frames=save_frames or bool(runtime_cfg.get("save_frames", False)),
            save_debug=save_debug or bool(runtime_cfg.get("save_debug", False)),
        ),
        models=ModelConfig(
            sam3_model_dir=str(models_cfg.get("sam3_model_dir")),
            yolo_detector=str(models_cfg.get("yolo_detector", "yolov8n.pt")),
            yolo_segmenter=str(models_cfg.get("yolo_segmenter", "yolov8n-seg.pt")),
        ),
        thresholds=ThresholdConfig(
            det_conf=float(thresholds_cfg.get("det_conf", 0.35)),
            min_box_area_ratio=float(thresholds_cfg.get("min_box_area_ratio", 0.0005)),
            max_box_area_ratio=float(thresholds_cfg.get("max_box_area_ratio", 0.8)),
            fallback_min_area=int(thresholds_cfg.get("fallback_min_area", 100)),
            fallback_iou_threshold=float(thresholds_cfg.get("fallback_iou_threshold", 0.2)),
            fallback_center_distance_ratio=float(
                thresholds_cfg.get("fallback_center_distance_ratio", 0.25)
            ),
            mask_threshold=float(thresholds_cfg.get("mask_threshold", 0.5)),
            area_ratio_spike=float(thresholds_cfg.get("area_ratio_spike", 4.0)),
        ),
        postprocess=PostprocessConfig(
            binary_threshold=float(post_cfg.get("binary_threshold", 0.5)),
            dilate_kernel=int(post_cfg.get("dilate_kernel", 7)),
            dilate_iterations=int(post_cfg.get("dilate_iterations", 1)),
            close_kernel=int(post_cfg.get("close_kernel", 7)),
            min_component_area=int(post_cfg.get("min_component_area", 64)),
        ),
        prompt=PromptConfig(
            target_text=target_text,
            yolo_class=yolo_class,
            prompt_box=prompt_box,
            all_instances=all_instances,
        ),
    )
