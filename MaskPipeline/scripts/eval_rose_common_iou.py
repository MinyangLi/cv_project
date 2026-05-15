from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import build_app_config, load_yaml_config
from src.io.video_io import H264VideoWriter, inspect_video, save_mask_png
from src.pipeline.video_mask_pipeline import run_video_mask_pipeline


CSV_COLUMNS = [
    "video_id",
    "part",
    "target_text",
    "prompt_box",
    "num_frames_pred",
    "num_frames_gt",
    "num_frames_evaluated",
    "JM",
    "JR",
    "mean_iou",
    "min_iou",
    "max_iou",
    "warnings",
    "pred_video",
]


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    target_text: str
    prompt_box: list[tuple[int, int, int, int]] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RoSE common Part 3 mask generation and IoU evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "rose_common_part3.yaml",
        help="Dataset config YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "IoU",
        help="Evaluation output root.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device for SAM3 tracker.")
    parser.add_argument("--force", action="store_true", help="Re-run prediction even if outputs already exist.")
    return parser


def load_eval_config(path: Path) -> tuple[dict, list[VideoSpec]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    videos = [
        VideoSpec(
            video_id=str(item["video_id"]),
            target_text=str(item["target_text"]),
            prompt_box=(
                [tuple(int(v) for v in box) for box in item.get("prompt_box", [])]
                if item.get("prompt_box") is not None
                else None
            ),
        )
        for item in raw.get("videos", [])
    ]
    if not videos:
        raise ValueError(f"No videos configured in {path}")
    return raw, videos


def read_first_nonempty_component_boxes(gt_video: Path, *, min_area: int) -> list[tuple[int, int, int, int]]:
    cap = cv2.VideoCapture(str(gt_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open GT video: {gt_video}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = (gray > 127).astype(np.uint8)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            boxes: list[tuple[int, int, int, int]] = []
            for label in range(1, num_labels):
                x, y, w, h, area = stats[label]
                if int(area) < min_area:
                    continue
                boxes.append((int(x), int(y), int(x + w), int(y + h)))
            if boxes:
                boxes.sort()
                return boxes
    finally:
        cap.release()

    raise RuntimeError(f"No non-empty GT mask components found in {gt_video}")


def read_video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Unable to read video size: {video_path}")
    return width, height


def scale_box(
    box: tuple[int, int, int, int],
    *,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if src_size == dst_size:
        return box
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    x1, y1, x2, y2 = box
    sx = dst_w / src_w
    sy = dst_h / src_h
    scaled = (
        max(0, min(dst_w - 1, int(round(x1 * sx)))),
        max(0, min(dst_h - 1, int(round(y1 * sy)))),
        max(1, min(dst_w, int(round(x2 * sx)))),
        max(1, min(dst_h, int(round(y2 * sy)))),
    )
    x1n, y1n, x2n, y2n = scaled
    if x2n <= x1n:
        x2n = min(dst_w, x1n + 1)
    if y2n <= y1n:
        y2n = min(dst_h, y1n + 1)
    return (x1n, y1n, x2n, y2n)


def load_mask_png(path: Path, *, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Failed to read mask png: {path}")
    mask = (mask > 127).astype(np.uint8) * 255
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def write_union_prediction_video(
    component_mask_dirs: list[Path],
    output_video: Path,
    pred_masks_dir: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
) -> None:
    pred_masks_dir.mkdir(parents=True, exist_ok=True)
    writer = H264VideoWriter(output_video, width=width, height=height, fps=fps)
    try:
        for frame_index in range(frame_count):
            merged = np.zeros((height, width), dtype=np.uint8)
            frame_name = f"{frame_index:05d}.png"
            for mask_dir in component_mask_dirs:
                merged = np.maximum(merged, load_mask_png(mask_dir / frame_name, shape=(height, width)))
            save_mask_png(pred_masks_dir / frame_name, merged)
            writer.write(cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR))
    finally:
        writer.close()


def evaluate_prediction(pred_video: Path, gt_video: Path) -> tuple[dict[str, float | int], list[str]]:
    pred_cap = cv2.VideoCapture(str(pred_video))
    gt_cap = cv2.VideoCapture(str(gt_video))
    if not pred_cap.isOpened():
        raise FileNotFoundError(f"Unable to open prediction video: {pred_video}")
    if not gt_cap.isOpened():
        raise FileNotFoundError(f"Unable to open GT video: {gt_video}")

    warnings: list[str] = []
    pred_frames = int(pred_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    gt_frames = int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pred_w = int(pred_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    pred_h = int(pred_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    gt_w = int(gt_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    gt_h = int(gt_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if pred_frames != gt_frames:
        warnings.append(f"frame_count_mismatch:pred={pred_frames},gt={gt_frames}")
    if (pred_w, pred_h) != (gt_w, gt_h):
        warnings.append(f"resolution_mismatch:pred={pred_w}x{pred_h},gt={gt_w}x{gt_h}")

    frame_limit = min(pred_frames, gt_frames)
    ious: list[float] = []
    try:
        for _ in range(frame_limit):
            ok_pred, pred_frame = pred_cap.read()
            ok_gt, gt_frame = gt_cap.read()
            if not (ok_pred and ok_gt):
                warnings.append("read_failure_before_frame_limit")
                break
            pred_gray = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY)
            gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
            pred_mask = (pred_gray > 127).astype(np.uint8)
            gt_mask = (gt_gray > 127).astype(np.uint8)
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            intersection = int(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
            union = int(np.logical_or(pred_mask > 0, gt_mask > 0).sum())
            iou = 1.0 if union == 0 else intersection / union
            ious.append(float(iou))
    finally:
        pred_cap.release()
        gt_cap.release()

    if not ious:
        raise RuntimeError(f"No frames evaluated for {pred_video}")

    mean_iou = float(np.mean(ious))
    return {
        "num_frames_pred": pred_frames,
        "num_frames_gt": gt_frames,
        "num_frames_evaluated": len(ious),
        "JM": mean_iou,
        "JR": float(np.mean(np.array(ious) > 0.5)),
        "mean_iou": mean_iou,
        "min_iou": float(np.min(ious)),
        "max_iou": float(np.max(ious)),
    }, warnings


def normalize_metric(value: float | int) -> float | int:
    if isinstance(value, float):
        return round(value, 6)
    return value


def infer_yolo_class(target_text: str) -> str | None:
    lowered = target_text.lower()
    for name in ("person", "car", "bus", "bicycle", "motorcycle", "truck"):
        if name in lowered:
            return name
    return None


def run_component_prediction(
    *,
    input_video: Path,
    target_text: str,
    prompt_box: tuple[int, int, int, int],
    output_video: Path,
    base_config_path: Path,
    device: str,
) -> None:
    raw_config = load_yaml_config(base_config_path)
    app_config = build_app_config(
        raw_config,
        target_text=target_text,
        yolo_class=infer_yolo_class(target_text),
        prompt_box=prompt_box,
        all_instances=False,
        device=device,
        enable_optical_flow=None,
        allow_yolo_only_fallback=False,
        max_frames=None,
        save_frames=True,
        save_debug=False,
    )
    run_video_mask_pipeline(input_video, output_video, app_config)


def build_overall_row(rows: list[dict]) -> dict:
    all_ious = [row["_frame_metrics"] for row in rows]
    flattened = [iou for metrics in all_ious for iou in metrics]
    warnings = [row["warnings"] for row in rows if row["warnings"]]
    return {
        "video_id": "overall",
        "part": rows[0]["part"] if rows else "part3",
        "target_text": "",
        "prompt_box": "",
        "num_frames_pred": int(sum(int(row["num_frames_pred"]) for row in rows)),
        "num_frames_gt": int(sum(int(row["num_frames_gt"]) for row in rows)),
        "num_frames_evaluated": int(sum(int(row["num_frames_evaluated"]) for row in rows)),
        "JM": round(float(np.mean(flattened)), 6),
        "JR": round(float(np.mean(np.array(flattened) > 0.5)), 6),
        "mean_iou": round(float(np.mean(flattened)), 6),
        "min_iou": round(float(np.min(flattened)), 6),
        "max_iou": round(float(np.max(flattened)), 6),
        "warnings": " | ".join(warnings),
        "pred_video": "",
    }


def evaluate_prediction_with_frames(pred_video: Path, gt_video: Path) -> tuple[dict[str, float | int], list[str], list[float]]:
    metrics, warnings = evaluate_prediction(pred_video, gt_video)
    pred_cap = cv2.VideoCapture(str(pred_video))
    gt_cap = cv2.VideoCapture(str(gt_video))
    frame_limit = min(
        int(pred_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    ious: list[float] = []
    try:
        for _ in range(frame_limit):
            ok_pred, pred_frame = pred_cap.read()
            ok_gt, gt_frame = gt_cap.read()
            if not (ok_pred and ok_gt):
                break
            pred_gray = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY)
            gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
            pred_mask = (pred_gray > 127).astype(np.uint8)
            gt_mask = (gt_gray > 127).astype(np.uint8)
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            intersection = int(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
            union = int(np.logical_or(pred_mask > 0, gt_mask > 0).sum())
            ious.append(1.0 if union == 0 else intersection / union)
    finally:
        pred_cap.release()
        gt_cap.release()
    return metrics, warnings, ious


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw_eval_config, video_specs = load_eval_config(args.config)
    dataset_root = Path(raw_eval_config["dataset_root"])
    base_config_path = Path(raw_eval_config.get("base_config", PROJECT_ROOT / "configs" / "mask_video.yaml"))
    part = str(raw_eval_config.get("part", "part3"))
    min_component_area = int(raw_eval_config.get("component_area_threshold", 64))

    part_root = args.output_root / part
    work_root = part_root / "work"
    pred_videos_root = part_root / "pred_videos"
    pred_masks_root = part_root / "pred_masks"
    pred_meta_root = part_root / "pred_meta"
    summary_path = part_root / "part3_common_summary.json"
    csv_path = part_root / "part3_common_metrics.csv"
    for path in (work_root, pred_videos_root, pred_masks_root, pred_meta_root):
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in video_specs:
        input_video = dataset_root / "Unedited" / f"{spec.video_id}.mp4"
        gt_video = dataset_root / "Masked" / f"{spec.video_id}.mp4"
        if spec.prompt_box is not None:
            prompt_boxes = spec.prompt_box
        else:
            gt_boxes = read_first_nonempty_component_boxes(gt_video, min_area=min_component_area)
            gt_size = read_video_size(gt_video)
            input_size = read_video_size(input_video)
            prompt_boxes = [scale_box(box, src_size=gt_size, dst_size=input_size) for box in gt_boxes]
        video_info = inspect_video(gt_video)

        logging.info("Processing %s with %d prompt boxes", spec.video_id, len(prompt_boxes))
        component_mask_dirs: list[Path] = []
        component_meta_paths: list[str] = []
        for component_index, prompt_box in enumerate(prompt_boxes, start=1):
            component_output = work_root / spec.video_id / f"component-{component_index:02d}_masked.mp4"
            component_masks_dir = component_output.parent / f"component-{component_index:02d}" / "masks"
            component_meta = component_output.parent / f"component-{component_index:02d}" / "meta.json"
            component_mask_dirs.append(component_masks_dir)
            component_meta_paths.append(str(component_meta))
            if args.force or not component_meta.exists():
                run_component_prediction(
                    input_video=input_video,
                    target_text=spec.target_text,
                    prompt_box=prompt_box,
                    output_video=component_output,
                    base_config_path=base_config_path,
                    device=args.device,
                )

        pred_video = pred_videos_root / f"{spec.video_id}_{part}_masked.mp4"
        pred_masks_dir = pred_masks_root / spec.video_id
        write_union_prediction_video(
            component_mask_dirs,
            pred_video,
            pred_masks_dir,
            frame_count=video_info.num_frames,
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
        )

        metrics, metric_warnings, frame_ious = evaluate_prediction_with_frames(pred_video, gt_video)
        warnings = metric_warnings.copy()
        if len(prompt_boxes) > 1:
            warnings.append(f"multi_component_union:{len(prompt_boxes)}")

        row = {
            "video_id": spec.video_id,
            "part": part,
            "target_text": spec.target_text,
            "prompt_box": json.dumps(prompt_boxes, ensure_ascii=False),
            "num_frames_pred": int(metrics["num_frames_pred"]),
            "num_frames_gt": int(metrics["num_frames_gt"]),
            "num_frames_evaluated": int(metrics["num_frames_evaluated"]),
            "JM": normalize_metric(float(metrics["JM"])),
            "JR": normalize_metric(float(metrics["JR"])),
            "mean_iou": normalize_metric(float(metrics["mean_iou"])),
            "min_iou": normalize_metric(float(metrics["min_iou"])),
            "max_iou": normalize_metric(float(metrics["max_iou"])),
            "warnings": " | ".join(warnings),
            "pred_video": str(pred_video),
            "_frame_metrics": frame_ious,
        }
        rows.append(row)

        meta_path = pred_meta_root / f"{spec.video_id}.json"
        meta_payload = {
            "video_id": spec.video_id,
            "part": part,
            "target_text": spec.target_text,
            "prompt_box": prompt_boxes,
            "pred_video": str(pred_video),
            "gt_video": str(gt_video),
            "component_meta": component_meta_paths,
            "metrics": {key: row[key] for key in CSV_COLUMNS if key in row},
            "warnings": warnings,
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, indent=2, ensure_ascii=False)

    overall_row = build_overall_row(rows)
    csv_rows = [{key: row[key] for key in CSV_COLUMNS} for row in rows] + [overall_row]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_payload = {
        "part": part,
        "dataset": str(dataset_root),
        "csv": str(csv_path),
        "videos": [{key: row[key] for key in CSV_COLUMNS} for row in rows],
        "overall": overall_row,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    logging.info("Wrote CSV: %s", csv_path)
    logging.info("Wrote summary: %s", summary_path)


if __name__ == "__main__":
    main()
