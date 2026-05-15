from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.video_io import H264VideoWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DAVIS-style image sequence dataset as MP4 videos for IoU evaluation.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/czy/cv_project/MaskPipeline/Input_video/Unedited"),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/home/czy/ROSE-Benchmark/Benchmark/common/Masked/1080p"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/czy/cv_project/MaskPipeline/outputs/IoU/Davis/dataset_videos"),
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--datasets", nargs="+", default=["bmx-trees", "tennis"])
    return parser


def sorted_image_paths(folder: Path, suffix: str) -> list[Path]:
    return sorted(folder.glob(f"*{suffix}"))


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_video_from_frames(frame_paths: list[Path], output_video: Path, *, fps: float, binary_mask: bool) -> None:
    if not frame_paths:
        raise RuntimeError(f"No frames for {output_video}")
    sample = read_gray(frame_paths[0]) if binary_mask else read_color(frame_paths[0])
    height, width = sample.shape[:2]
    writer = H264VideoWriter(output_video, width=width, height=height, fps=fps)
    try:
        for path in frame_paths:
            if binary_mask:
                gray = read_gray(path)
                mask = ((gray > 127).astype(np.uint8) * 255)
                writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            else:
                writer.write(read_color(path))
    finally:
        writer.close()


def main() -> None:
    args = build_parser().parse_args()
    unedited_root = args.output_root / "Unedited"
    masked_root = args.output_root / "Masked"
    unedited_root.mkdir(parents=True, exist_ok=True)
    masked_root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        input_dir = args.input_root / name
        gt_dir = args.gt_root / name
        input_frames = sorted_image_paths(input_dir, ".jpg")
        gt_frames = sorted_image_paths(gt_dir, ".png")
        if len(input_frames) != len(gt_frames):
            raise RuntimeError(f"Frame count mismatch for {name}: input={len(input_frames)}, gt={len(gt_frames)}")
        write_video_from_frames(input_frames, unedited_root / f"{name}.mp4", fps=args.fps, binary_mask=False)
        write_video_from_frames(gt_frames, masked_root / f"{name}.mp4", fps=args.fps, binary_mask=True)
        print(f"[prepare_davis_eval_dataset] wrote {name}: {len(input_frames)} frames", flush=True)


if __name__ == "__main__":
    main()
