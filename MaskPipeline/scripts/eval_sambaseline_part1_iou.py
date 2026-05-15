from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from eval_sambaseline_sam2_iou import evaluate_prediction, load_ours_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMBASELINE_ROOT = PROJECT_ROOT / "sambaseline"

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM baseline Part 1 style pipeline and compare IoU metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sambaseline_part1_3478.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "IoU" / "sambaseline_part1",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def load_config(path: Path) -> tuple[dict, list[VideoSpec]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs = [
        VideoSpec(video_id=str(item["video_id"]), target_text=str(item["target_text"]))
        for item in raw.get("videos", [])
    ]
    if not specs:
        raise ValueError(f"No videos in {path}")
    return raw, specs


def run_part1_pipeline(input_video: Path, output_dir: Path, device: str) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    cmd = [
        "python",
        "part2_pipeline.py",
        "--input",
        str(input_video),
        "--output-dir",
        str(output_dir),
        "--mask-backend",
        "open_choice",
        "--inpaint-backend",
        "opencv",
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=SAMBASELINE_ROOT, check=True)


def main() -> None:
    args = build_parser().parse_args()
    raw, specs = load_config(args.config)
    dataset_root = Path(raw["dataset_root"])
    method = str(raw.get("method", "sambaseline_part1"))
    part = str(raw.get("part", "part1"))

    pred_runs_root = args.output_root / "runs"
    pred_videos_root = args.output_root / "pred_videos"
    pred_meta_root = args.output_root / "pred_meta"
    for path in (pred_runs_root, pred_videos_root, pred_meta_root):
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    compare_rows: list[dict] = []
    for spec in specs:
        input_video = dataset_root / "Unedited" / f"{spec.video_id}.mp4"
        gt_video = dataset_root / "Masked" / f"{spec.video_id}.mp4"
        run_dir = pred_runs_root / spec.video_id
        pred_video = pred_videos_root / f"{spec.video_id}_{method}_masked.mp4"

        print(f"[sambaseline_part1] start video={spec.video_id} target={spec.target_text}", flush=True)
        run_part1_pipeline(input_video, run_dir, args.device)
        generated_mask_video = run_dir / "part2_masks.mp4"
        if not generated_mask_video.exists():
            raise FileNotFoundError(f"Missing mask video: {generated_mask_video}")
        shutil.copy2(generated_mask_video, pred_video)
        metrics, warnings, _ = evaluate_prediction(pred_video, gt_video)
        row = {
            "method": method,
            "video_id": spec.video_id,
            "part": part,
            "target_text": spec.target_text,
            "prompt_box": "",
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
                "sambaseline_part1_JM": row["JM"],
                "sambaseline_part1_JR": row["JR"],
                "delta_JM": None if ours.get("JM") is None else round(float(row["JM"]) - float(ours["JM"]), 6),
                "delta_JR": None if ours.get("JR") is None else round(float(row["JR"]) - float(ours["JR"]), 6),
            }
        )
        print(f"[sambaseline_part1] done video={spec.video_id} JM={row['JM']} JR={row['JR']}", flush=True)

    csv_path = args.output_root / "sambaseline_part1_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    compare_path = args.output_root / "compare_with_ours_3478.csv"
    compare_fields = [
        "video_id",
        "target_text",
        "ours_JM",
        "ours_JR",
        "sambaseline_part1_JM",
        "sambaseline_part1_JR",
        "delta_JM",
        "delta_JR",
    ]
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
    (args.output_root / "sambaseline_part1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[sambaseline_part1] wrote summary {args.output_root / 'sambaseline_part1_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
