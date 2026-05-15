#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DAVIS image/mask frame folders to fixed-length videos."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("DAVIS"),
        help="Dataset root containing images/ and test_masks/.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Subdirectory name for RGB frame sequences.",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default="test_masks",
        help="Subdirectory name for mask frame sequences.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Default: <root>/videos",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=90,
        help="Target frame count for each output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Target output width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Target output height.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".mp4",
        help="Output video file suffix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output videos if they already exist.",
    )
    return parser.parse_args()


def list_sequence_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        raise FileNotFoundError(f"Directory not found: {parent}")
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def list_frame_paths(seq_dir: Path, exts: tuple[str, ...]) -> list[Path]:
    frames = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(frames)


def build_sample_indices(src_len: int, target_len: int) -> np.ndarray:
    if src_len <= 0:
        raise ValueError("Source sequence has no frames.")
    if target_len <= 0:
        raise ValueError("Target frame count must be positive.")
    if src_len == 1:
        return np.zeros(target_len, dtype=np.int64)
    return np.rint(np.linspace(0, src_len - 1, target_len)).astype(np.int64)


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def write_video_from_sequence(
    seq_dir: Path,
    out_path: Path,
    target_frames: int,
    size: tuple[int, int],
    fps: float,
    image_exts: tuple[str, ...],
    interpolation: int,
) -> int:
    frame_paths = list_frame_paths(seq_dir, image_exts)
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in sequence: {seq_dir}")

    indices = build_sample_indices(len(frame_paths), target_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {out_path}")

    written = 0
    try:
        for idx in indices:
            frame = cv2.imread(str(frame_paths[int(idx)]), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_paths[int(idx)]}")
            frame = ensure_bgr(frame)
            frame = cv2.resize(frame, size, interpolation=interpolation)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    return written


def convert_group(
    in_parent: Path,
    out_parent: Path,
    target_frames: int,
    size: tuple[int, int],
    fps: float,
    suffix: str,
    overwrite: bool,
    image_exts: tuple[str, ...],
    interpolation: int,
) -> None:
    seq_dirs = list_sequence_dirs(in_parent)
    print(f"[INFO] Processing {in_parent} -> {out_parent}, sequences={len(seq_dirs)}")

    ok_count = 0
    skip_count = 0
    for seq_dir in seq_dirs:
        out_path = out_parent / f"{seq_dir.name}{suffix}"
        if out_path.exists() and not overwrite:
            print(f"[SKIP] Exists: {out_path}")
            skip_count += 1
            continue

        written = write_video_from_sequence(
            seq_dir=seq_dir,
            out_path=out_path,
            target_frames=target_frames,
            size=size,
            fps=fps,
            image_exts=image_exts,
            interpolation=interpolation,
        )
        print(f"[OK] {seq_dir.name}: {written} frames -> {out_path.name}")
        ok_count += 1

    print(f"[DONE] {in_parent.name}: converted={ok_count}, skipped={skip_count}")


def main() -> int:
    args = parse_args()

    root = args.root.resolve()
    out_root = (args.out_root if args.out_root is not None else root / "videos").resolve()

    images_in = root / args.images_dir
    masks_in = root / args.masks_dir
    images_out = out_root / "images"
    masks_out = out_root / "masks"

    size = (args.width, args.height)

    convert_group(
        in_parent=images_in,
        out_parent=images_out,
        target_frames=args.num_frames,
        size=size,
        fps=args.fps,
        suffix=args.suffix,
        overwrite=args.overwrite,
        image_exts=(".jpg", ".jpeg", ".png", ".bmp"),
        interpolation=cv2.INTER_CUBIC,
    )

    convert_group(
        in_parent=masks_in,
        out_parent=masks_out,
        target_frames=args.num_frames,
        size=size,
        fps=args.fps,
        suffix=args.suffix,
        overwrite=args.overwrite,
        image_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        interpolation=cv2.INTER_NEAREST,
    )

    print("[ALL DONE] Conversion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
