from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sambaseline.sam2_auto_mask import postprocess_mask  # type: ignore
from src.io.video_io import H264VideoWriter

LOCAL_SAM2_CONFIG = "configs/sam2/sam2_hiera_s.yaml"
LOCAL_SAM2_CKPT = "/data/sam2/sam2_hiera_small.pt"


CSV_COLUMNS = [
    "method",
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
    prompt_box: tuple[int, int, int, int] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM baseline SAM2 on explicit boxes and compare IoU metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sambaseline_sam2_3478.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "IoU" / "sambaseline_sam2",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def load_config(path: Path) -> tuple[dict, list[VideoSpec]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs = [
        VideoSpec(
            video_id=str(item["video_id"]),
            target_text=str(item["target_text"]),
            prompt_box=(tuple(int(v) for v in item["prompt_box"]) if item.get("prompt_box") is not None else None),
        )
        for item in raw.get("videos", [])
    ]
    if not specs:
        raise ValueError(f"No videos in {path}")
    return raw, specs


def read_first_nonempty_main_box(gt_video: Path, *, min_area: int) -> tuple[int, int, int, int]:
    cap = cv2.VideoCapture(str(gt_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open GT video: {gt_video}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = (gray > 127).astype(np.uint8)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            best_box = None
            best_area = -1
            for label in range(1, num_labels):
                x, y, w, h, area = stats[label]
                if int(area) < min_area:
                    continue
                if int(area) > best_area:
                    best_area = int(area)
                    best_box = (int(x), int(y), int(x + w), int(y + h))
            if best_box is not None:
                return best_box
    finally:
        cap.release()
    raise RuntimeError(f"No non-empty GT component found in {gt_video}")


def read_video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video size from {video_path}")
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


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    out = (mask > 0).astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def extract_video_to_jpg_frames(input_video: Path, frame_dir: Path) -> tuple[int, int, int]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("*.jpg"))
    if existing:
        probe = cv2.imread(str(existing[0]))
        cap = cv2.VideoCapture(str(input_video))
        if probe is not None and cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if frame_count == len(existing):
                return frame_count, width, height
        else:
            cap.release()

    for old in frame_dir.glob("*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {input_video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = frame_dir / f"{idx:05d}.jpg"
            if not cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise RuntimeError(f"Failed to write frame {out_path}")
            idx += 1
    finally:
        cap.release()

    return idx or frame_count, width, height


def run_sam2_with_box(
    *,
    input_video: Path,
    prompt_box: tuple[int, int, int, int],
    frame_dir: Path,
    mask_dir: Path,
    pred_video: Path,
    device: str,
) -> tuple[int, int, int]:
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore

    mask_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(
        config_file=LOCAL_SAM2_CONFIG,
        ckpt_path=LOCAL_SAM2_CKPT,
        device=device,
    )
    frame_count, width, height = extract_video_to_jpg_frames(input_video, frame_dir)
    state = predictor.init_state(str(frame_dir))

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    cap.release()

    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.lower().startswith("cuda") else torch.cpu.amp.autocast()
    mask_map: dict[int, np.ndarray] = {}
    with torch.inference_mode(), amp_ctx:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=np.asarray(prompt_box, dtype=np.float32),
        )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            combined = np.zeros((height, width), dtype=bool)
            for j in range(len(out_obj_ids)):
                m = np.squeeze(out_mask_logits[j].detach().cpu().numpy())
                combined = np.logical_or(combined, m > 0.0)
            mask_map[int(out_frame_idx)] = combined.astype(np.uint8) * 255

    writer = H264VideoWriter(pred_video, width=width, height=height, fps=fps)
    try:
        for idx in range(frame_count):
            mask = mask_map.get(idx, np.zeros((height, width), dtype=np.uint8))
            mask = postprocess_mask(mask, dilate_ksize=11, close_ksize=7, open_ksize=0, min_area=80)
            save_mask_png(mask_dir / f"mask_{idx:05d}.png", mask)
            writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    finally:
        writer.close()
    return frame_count, width, height


def evaluate_prediction(pred_video: Path, gt_video: Path) -> tuple[dict[str, float | int], list[str], list[float]]:
    pred_cap = cv2.VideoCapture(str(pred_video))
    gt_cap = cv2.VideoCapture(str(gt_video))
    if not pred_cap.isOpened() or not gt_cap.isOpened():
        raise RuntimeError(f"Cannot open prediction or GT: {pred_video} | {gt_video}")

    warnings: list[str] = []
    pred_frames = int(pred_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    gt_frames = int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_limit = min(pred_frames, gt_frames)
    if pred_frames != gt_frames:
        warnings.append(f"frame_count_mismatch:pred={pred_frames},gt={gt_frames}")

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
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            intersection = int(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
            union = int(np.logical_or(pred_mask > 0, gt_mask > 0).sum())
            ious.append(1.0 if union == 0 else intersection / union)
    finally:
        pred_cap.release()
        gt_cap.release()

    mean_iou = float(np.mean(ious))
    metrics = {
        "num_frames_pred": pred_frames,
        "num_frames_gt": gt_frames,
        "num_frames_evaluated": len(ious),
        "JM": round(mean_iou, 6),
        "JR": round(float(np.mean(np.array(ious) > 0.5)), 6),
        "mean_iou": round(mean_iou, 6),
        "min_iou": round(float(np.min(ious)), 6),
        "max_iou": round(float(np.max(ious)), 6),
    }
    return metrics, warnings, ious


def load_ours_metrics(video_id: str) -> dict:
    path = PROJECT_ROOT / "outputs" / "IoU" / "part3" / "pred_meta" / f"{video_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("metrics", {})


def main() -> None:
    args = build_parser().parse_args()
    raw, specs = load_config(args.config)
    dataset_root = Path(raw["dataset_root"])
    method = str(raw.get("method", "sambaseline_sam2"))
    part = str(raw.get("part", "part2"))
    min_component_area = int(raw.get("component_area_threshold", 512))

    pred_videos_root = args.output_root / "pred_videos"
    pred_masks_root = args.output_root / "pred_masks"
    pred_meta_root = args.output_root / "pred_meta"
    cache_frames_root = args.output_root / "cache_frames"
    for path in (pred_videos_root, pred_masks_root, pred_meta_root, cache_frames_root):
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    compare_rows: list[dict] = []
    for spec in specs:
        input_video = dataset_root / "Unedited" / f"{spec.video_id}.mp4"
        gt_video = dataset_root / "Masked" / f"{spec.video_id}.mp4"
        pred_video = pred_videos_root / f"{spec.video_id}_{method}_masked.mp4"
        mask_dir = pred_masks_root / spec.video_id
        frame_dir = cache_frames_root / spec.video_id
        if spec.prompt_box is not None:
            prompt_box = spec.prompt_box
        else:
            gt_box = read_first_nonempty_main_box(gt_video, min_area=min_component_area)
            gt_size = read_video_size(gt_video)
            input_size = read_video_size(input_video)
            prompt_box = scale_box(gt_box, src_size=gt_size, dst_size=input_size)

        print(
            f"[sambaseline_sam2] start video={spec.video_id} target={spec.target_text} box={list(prompt_box)}",
            flush=True,
        )

        run_sam2_with_box(
            input_video=input_video,
            prompt_box=prompt_box,
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            pred_video=pred_video,
            device=args.device,
        )
        print(f"[sambaseline_sam2] masks ready video={spec.video_id}", flush=True)
        metrics, warnings, _ = evaluate_prediction(pred_video, gt_video)
        row = {
            "method": method,
            "video_id": spec.video_id,
            "part": part,
            "target_text": spec.target_text,
            "prompt_box": json.dumps(list(prompt_box), ensure_ascii=False),
            "num_frames_pred": int(metrics["num_frames_pred"]),
            "num_frames_gt": int(metrics["num_frames_gt"]),
            "num_frames_evaluated": int(metrics["num_frames_evaluated"]),
            "JM": metrics["JM"],
            "JR": metrics["JR"],
            "mean_iou": metrics["mean_iou"],
            "min_iou": metrics["min_iou"],
            "max_iou": metrics["max_iou"],
            "warnings": " | ".join(warnings),
            "pred_video": str(pred_video),
        }
        rows.append(row)
        (pred_meta_root / f"{spec.video_id}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

        ours = load_ours_metrics(spec.video_id)
        compare_rows.append(
            {
                "video_id": spec.video_id,
                "target_text": spec.target_text,
                "ours_JM": ours.get("JM"),
                "ours_JR": ours.get("JR"),
                "sambaseline_JM": row["JM"],
                "sambaseline_JR": row["JR"],
                "delta_JM": None if ours.get("JM") is None else round(float(row["JM"]) - float(ours["JM"]), 6),
                "delta_JR": None if ours.get("JR") is None else round(float(row["JR"]) - float(ours["JR"]), 6),
            }
        )
        print(
            f"[sambaseline_sam2] done video={spec.video_id} JM={row['JM']} JR={row['JR']}",
            flush=True,
        )

    csv_path = args.output_root / "sambaseline_sam2_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    compare_path = args.output_root / "compare_with_ours_3478.csv"
    compare_fields = ["video_id", "target_text", "ours_JM", "ours_JR", "sambaseline_JM", "sambaseline_JR", "delta_JM", "delta_JR"]
    with compare_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compare_fields)
        writer.writeheader()
        writer.writerows(compare_rows)

    summary = {
        "method": method,
        "csv": str(csv_path),
        "compare_csv": str(compare_path),
        "videos": rows,
    }
    (args.output_root / "sambaseline_sam2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[sambaseline_sam2] wrote summary {args.output_root / 'sambaseline_sam2_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
