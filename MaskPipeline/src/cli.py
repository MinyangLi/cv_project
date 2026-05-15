from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import build_app_config, load_yaml_config


def parse_prompt_box(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("prompt-box must be 'x1,y1,x2,y2'")
    try:
        coords = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prompt-box must contain integers") from exc
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("prompt-box must satisfy x2>x1 and y2>y1")
    return coords


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video-level object mask pipeline using SAM3 + YOLO fallback.")
    parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    parser.add_argument("--output", type=Path, default=None, help="Output masked video path.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path.")
    parser.add_argument("--target-text", type=str, default=None, help="SAM3 text prompt, e.g. 'the man in white shirt'.")
    parser.add_argument("--yolo-class", type=str, default=None, help="Optional YOLO class for fallback.")
    parser.add_argument("--target-class", type=str, default=None, help="Deprecated alias; maps to both target-text and yolo-class.")
    parser.add_argument("--prompt-box", type=parse_prompt_box, default=None, help="Single-instance box prompt: x1,y1,x2,y2")
    parser.add_argument("--all-instances", action="store_true", help="Return all matching instances instead of one tracked instance.")
    parser.add_argument("--max-frames", type=int, default=None, help="Only process the first N frames.")
    parser.add_argument("--save-frames", action="store_true", help="Save binary mask PNG frames.")
    parser.add_argument("--save-debug", action="store_true", help="Save overlay PNG frames.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. cpu or cuda.")
    parser.add_argument("--enable-optical-flow", action="store_true", help="Enable optical-flow refinement.")
    parser.add_argument("--disable-optical-flow", action="store_true", help="Disable optical-flow refinement.")
    parser.add_argument("--allow-yolo-only-fallback", action="store_true", help="Allow running without SAM3 by using YOLO-only fallback.")
    return parser


def resolve_cli_config(args: argparse.Namespace):
    raw_config = load_yaml_config(args.config)
    target_text = args.target_text
    yolo_class = args.yolo_class
    if args.target_class:
        target_text = target_text or args.target_class
        yolo_class = yolo_class or args.target_class
    if not target_text:
        raise ValueError("target-text is required. You may use --target-class as a backward-compatible alias.")
    if args.enable_optical_flow and args.disable_optical_flow:
        raise ValueError("enable-optical-flow and disable-optical-flow cannot both be set.")
    enable_optical_flow = True if args.enable_optical_flow else False if args.disable_optical_flow else None

    app_config = build_app_config(
        raw_config,
        target_text=target_text,
        yolo_class=yolo_class,
        prompt_box=args.prompt_box,
        all_instances=args.all_instances,
        device=args.device,
        enable_optical_flow=enable_optical_flow,
        allow_yolo_only_fallback=args.allow_yolo_only_fallback,
        max_frames=args.max_frames,
        save_frames=args.save_frames,
        save_debug=args.save_debug,
    )
    return app_config
