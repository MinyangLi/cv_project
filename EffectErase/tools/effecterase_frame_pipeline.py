#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame sequence <-> video helpers for EffectErase remove inference.

- prepare: resample arbitrary-length RGB + mask folders to fixed T=4n+1 (default 81),
  write two MP4s (fps = num_frames / duration_sec, default 3s -> 27fps).
- extract: dump video frames to PNG.
- run-all: prepare -> infer_remove_wan.py -> extract (+ optional resample to source count).
- prepare-gt: one RGB frame folder -> same temporal resample as inference input -> PNGs (default 81).
- prepare-gt-batch: under images_root, each subfolder (e.g. bear/) -> out_root/<name>/ (81 PNGs).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _natural_sort_key(name: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def list_image_files(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")
    files = [d / f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)]
    files.sort(key=lambda p: _natural_sort_key(p.name))
    return files


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if arr.ndim == 3:
        gray = arr.max(axis=2)
    else:
        gray = arr
    return gray.astype(np.uint8)


def temporal_resample_rgb(frames: np.ndarray, t_out: int) -> np.ndarray:
    """frames: (N,H,W,3) uint8 -> (t_out,H,W,3) uint8, linear blend in time."""
    n, h, w, c = frames.shape
    if c != 3:
        raise ValueError(f"Expected RGB NHW3, got {frames.shape}")
    if n == 0:
        raise ValueError("empty frames")
    if t_out < 1:
        raise ValueError("t_out must be >= 1")
    f = frames.astype(np.float32)
    if n == 1:
        return np.broadcast_to(f[0], (t_out, h, w, 3)).round().clip(0, 255).astype(np.uint8)
    out = np.empty((t_out, h, w, 3), dtype=np.float32)
    for k in range(t_out):
        t = k * (n - 1) / (t_out - 1) if t_out > 1 else 0.0
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, n - 1)
        a = t - i0
        out[k] = (1.0 - a) * f[i0] + a * f[i1]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def temporal_resample_mask_nearest(masks: np.ndarray, t_out: int) -> np.ndarray:
    """masks: (N,H,W) uint8 -> (t_out,H,W) uint8, nearest frame index."""
    n = masks.shape[0]
    if n == 0:
        raise ValueError("empty masks")
    if t_out < 1:
        raise ValueError("t_out must be >= 1")
    if n == 1:
        return np.broadcast_to(masks[0], (t_out, *masks.shape[1:])).copy()
    idx = np.linspace(0, n - 1, t_out)
    idx = np.round(idx).astype(np.int32).clip(0, n - 1)
    return masks[idx]


def write_mp4_bgr(frames_bgr: np.ndarray, out_path: Path, fps: float) -> None:
    """frames_bgr: (T,H,W,3) uint8 BGR."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w = frames_bgr.shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")
    try:
        for i in range(t):
            writer.write(frames_bgr[i])
    finally:
        writer.release()


def prepare_videos_from_frame_dirs(
    rgb_dir: str | Path,
    mask_dir: str | Path,
    out_fg_mp4: str | Path,
    out_mask_mp4: str | Path,
    num_frames: int = 81,
    duration_sec: float = 3.0,
) -> tuple[int, int, float]:
    """
    Returns (source_frame_count, num_frames_written, fps).
    """
    if num_frames % 4 != 1:
        raise ValueError(
            f"num_frames must satisfy num_frames % 4 == 1 (Wan), got {num_frames}"
        )
    rgb_paths = list_image_files(rgb_dir)
    mask_paths = list_image_files(mask_dir)
    if len(rgb_paths) != len(mask_paths):
        raise ValueError(
            f"RGB count {len(rgb_paths)} != mask count {len(mask_paths)} "
            f"under {rgb_dir} vs {mask_dir}"
        )
    if len(rgb_paths) == 0:
        raise ValueError(f"No images in {rgb_dir}")

    rgbs = [_read_rgb(p) for p in rgb_paths]
    masks = [_read_mask_gray(p) for p in mask_paths]
    h0, w0 = rgbs[0].shape[:2]
    aligned_masks = []
    for m in masks:
        if m.shape[:2] != (h0, w0):
            m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
        aligned_masks.append(m)
    rgb_stack = np.stack(rgbs, axis=0)
    mask_stack = np.stack(aligned_masks, axis=0)

    n_src = rgb_stack.shape[0]
    rgb_r = temporal_resample_rgb(rgb_stack, num_frames)
    mask_r = temporal_resample_mask_nearest(mask_stack, num_frames)
    mask_rgb = np.stack([mask_r, mask_r, mask_r], axis=-1)

    fps = num_frames / float(duration_sec)
    fg_bgr = np.stack(
        [cv2.cvtColor(rgb_r[i], cv2.COLOR_RGB2BGR) for i in range(num_frames)]
    )
    mask_bgr = np.stack(
        [cv2.cvtColor(mask_rgb[i], cv2.COLOR_RGB2BGR) for i in range(num_frames)]
    )

    write_mp4_bgr(fg_bgr, Path(out_fg_mp4), fps)
    write_mp4_bgr(mask_bgr, Path(out_mask_mp4), fps)
    print(
        f"[prepare] source_frames={n_src} -> written={num_frames} "
        f"fps={fps:.4f} ({duration_sec}s) ->\n  {out_fg_mp4}\n  {out_mask_mp4}"
    )
    return n_src, num_frames, fps


def extract_video_to_frames(
    video_path: str | Path,
    out_dir: str | Path,
    name_pattern: str = "{:05d}.png",
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            path = out_dir / name_pattern.format(n)
            cv2.imwrite(str(path), frame)
            n += 1
    finally:
        cap.release()
    print(f"[extract] wrote {n} frames to {out_dir}")
    return n


def resample_rgb_folder_to_pngs(
    src_dir: str | Path,
    out_dir: str | Path,
    num_frames: int = 81,
    name_pattern: str = "{:05d}.png",
) -> int:
    """
    Load all RGB frames from src_dir (sorted), apply the same temporal_resample_rgb
    as prepare_videos_from_frame_dirs, write num_frames PNGs to out_dir.
    Returns num_frames written.
    """
    if num_frames % 4 != 1:
        raise ValueError(
            f"num_frames must satisfy num_frames % 4 == 1 (Wan), got {num_frames}"
        )
    paths = list_image_files(src_dir)
    if not paths:
        raise ValueError(f"No images in {src_dir}")
    rgbs = [_read_rgb(p) for p in paths]
    h0, w0 = rgbs[0].shape[:2]
    for i, im in enumerate(rgbs):
        if im.shape[:2] != (h0, w0):
            raise ValueError(
                f"Frame size mismatch in {src_dir}: {paths[0].name} {h0}x{w0} vs "
                f"{paths[i].name} {im.shape[0]}x{im.shape[1]}"
            )
    stack = np.stack(rgbs, axis=0)
    n_src = stack.shape[0]
    out_rgb = temporal_resample_rgb(stack, num_frames)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        bgr = cv2.cvtColor(out_rgb[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / name_pattern.format(i)), bgr)
    print(
        f"[prepare-gt] {src_dir}: source_frames={n_src} -> {num_frames} PNGs -> {out_dir}"
    )
    return num_frames


def batch_resample_gt_images(
    images_root: str | Path,
    out_root: str | Path,
    num_frames: int = 81,
    sequences: list[str] | None = None,
) -> None:
    """
    For each subdirectory of images_root (e.g. bear/), write resampled GT frames
    to out_root/<subdir>/.
    """
    images_root = Path(images_root)
    out_root = Path(out_root)
    if not images_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {images_root}")
    want = None
    if sequences:
        want = {s.strip() for s in sequences if s.strip()}
    subs = sorted(
        p for p in images_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if want is not None:
        subs = [p for p in subs if p.name in want]
        missing = want - {p.name for p in subs}
        if missing:
            print(f"[WARN] sequences not found under {images_root}: {sorted(missing)}")
    if not subs:
        print(f"[WARN] no sequence folders to process under {images_root}")
        return
    for seq_dir in subs:
        try:
            resample_rgb_folder_to_pngs(
                seq_dir, out_root / seq_dir.name, num_frames=num_frames
            )
        except ValueError as e:
            print(f"[SKIP] {seq_dir.name}: {e}")


def cmd_prepare_gt(args: argparse.Namespace) -> None:
    resample_rgb_folder_to_pngs(
        args.src_dir,
        args.out_dir,
        num_frames=args.num_frames,
        name_pattern=args.pattern,
    )


def cmd_prepare_gt_batch(args: argparse.Namespace) -> None:
    seqs: list[str] | None = None
    raw = getattr(args, "sequences", "") or ""
    if raw.strip():
        seqs = [x.strip() for x in raw.split(",") if x.strip()]
    batch_resample_gt_images(
        args.images_root,
        args.out_root,
        num_frames=args.num_frames,
        sequences=seqs,
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    prepare_videos_from_frame_dirs(
        args.rgb_dir,
        args.mask_dir,
        args.out_fg_mp4,
        args.out_mask_mp4,
        num_frames=args.num_frames,
        duration_sec=args.duration_sec,
    )


def cmd_extract(args: argparse.Namespace) -> None:
    extract_video_to_frames(args.video, args.out_dir, args.pattern)


def cmd_run_all(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fg_mp4 = work / "fg_bg_resampled.mp4"
    mask_mp4 = work / "mask_resampled.mp4"
    out_mp4 = Path(args.output_video)

    n_src, _, _ = prepare_videos_from_frame_dirs(
        args.rgb_dir,
        args.mask_dir,
        fg_mp4,
        mask_mp4,
        num_frames=args.num_frames,
        duration_sec=args.duration_sec,
    )

    infer_py = root / "examples" / "remove_wan" / "infer_remove_wan.py"
    cmd = [
        sys.executable,
        str(infer_py),
        "--fg_bg_path",
        str(fg_mp4.resolve()),
        "--mask_path",
        str(mask_mp4.resolve()),
        "--output_path",
        str(out_mp4.resolve()),
        "--text_encoder_path",
        args.text_encoder_path,
        "--vae_path",
        args.vae_path,
        "--dit_path",
        args.dit_path,
        "--image_encoder_path",
        args.image_encoder_path,
        "--pretrained_lora_path",
        args.pretrained_lora_path,
        "--num_frames",
        str(args.num_frames),
        "--frame_interval",
        "1",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--seed",
        str(args.seed),
        "--cfg",
        str(args.cfg),
        "--num_inference_steps",
        str(args.num_inference_steps),
    ]
    if args.tiled:
        cmd.append("--tiled")
    if args.use_teacache:
        cmd.append("--use_teacache")
    extra = getattr(args, "infer_extra", None) or []
    cmd.extend(extra)

    print("[run-all] infer:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)

    out_dir = Path(args.output_frames_dir)
    if args.resample_output_to_source:
        tmp_dir = work / "remove_frames_native"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        n_out = extract_video_to_frames(out_mp4, tmp_dir)
        if n_out != args.num_frames:
            print(
                f"[WARN] expected {args.num_frames} frames from model video, got {n_out}"
            )
        stack = []
        for i in range(n_out):
            p = tmp_dir / f"{i:05d}.png"
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Missing frame {p}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stack.append(rgb)
        vid_rgb = np.stack(stack, axis=0)
        resampled = temporal_resample_rgb(vid_rgb, n_src)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_src):
            bgr = cv2.cvtColor(resampled[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), bgr)
        print(f"[run-all] resampled model output -> {n_src} frames in {out_dir}")
    else:
        extract_video_to_frames(out_mp4, out_dir)

    if not args.keep_temp:
        for p in (fg_mp4, mask_mp4):
            if p.is_file():
                p.unlink()
        if args.resample_output_to_source:
            tdir = work / "remove_frames_native"
            if tdir.is_dir():
                for c in tdir.glob("*.png"):
                    c.unlink()
                try:
                    tdir.rmdir()
                except OSError:
                    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EffectErase frame <-> video pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prepare", help="Frame folders -> two resampled MP4s")
    pr.add_argument("--rgb_dir", type=str, required=True)
    pr.add_argument("--mask_dir", type=str, required=True)
    pr.add_argument("--out_fg_mp4", type=str, required=True)
    pr.add_argument("--out_mask_mp4", type=str, required=True)
    pr.add_argument("--num_frames", type=int, default=81)
    pr.add_argument("--duration_sec", type=float, default=3.0)
    pr.set_defaults(func=cmd_prepare)

    ex = sub.add_parser("extract", help="Video -> numbered PNG frames")
    ex.add_argument("--video", type=str, required=True)
    ex.add_argument("--out_dir", type=str, required=True)
    ex.add_argument("--pattern", type=str, default="{:05d}.png")
    ex.set_defaults(func=cmd_extract)

    pg = sub.add_parser(
        "prepare-gt",
        help="One RGB frame folder -> temporally resampled GT PNGs (same as infer input)",
    )
    pg.add_argument("--src_dir", type=str, required=True)
    pg.add_argument("--out_dir", type=str, required=True)
    pg.add_argument("--num_frames", type=int, default=81)
    pg.add_argument("--pattern", type=str, default="{:05d}.png")
    pg.set_defaults(func=cmd_prepare_gt)

    pgb = sub.add_parser(
        "prepare-gt-batch",
        help="images_root/<seq>/* -> out_root/<seq>/*.png (each seq resampled to num_frames)",
    )
    pgb.add_argument("--images_root", type=str, required=True)
    pgb.add_argument("--out_root", type=str, required=True)
    pgb.add_argument("--num_frames", type=int, default=81)
    pgb.add_argument(
        "--sequences",
        type=str,
        default="",
        help="Comma-separated folder names under images_root (default: all subdirs)",
    )
    pgb.set_defaults(func=cmd_prepare_gt_batch)

    root = Path(__file__).resolve().parents[1]
    d_t5 = root / "Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth"
    d_vae = root / "Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth"
    d_dit = root / "Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"
    d_clip = (
        root
        / "Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    d_lora = root / "EffectErase.ckpt"

    ra = sub.add_parser(
        "run-all",
        help="prepare temp MP4s, run infer_remove_wan.py, extract PNGs",
    )
    ra.add_argument("--rgb_dir", type=str, required=True)
    ra.add_argument("--mask_dir", type=str, required=True)
    ra.add_argument("--output_frames_dir", type=str, required=True)
    ra.add_argument(
        "--output_video",
        type=str,
        default="",
        help="Save remove result MP4 here (default: work_dir/remove_out.mp4)",
    )
    ra.add_argument(
        "--work_dir",
        type=str,
        default="",
        help="Temp dir (default: <repo>/.effecterase_frame_work)",
    )
    ra.add_argument("--num_frames", type=int, default=81)
    ra.add_argument("--duration_sec", type=float, default=3.0)
    ra.add_argument("--height", type=int, default=480)
    ra.add_argument("--width", type=int, default=832)
    ra.add_argument("--seed", type=int, default=2025)
    ra.add_argument("--cfg", type=float, default=1.0)
    ra.add_argument("--num_inference_steps", type=int, default=50)
    ra.add_argument("--tiled", action="store_true")
    ra.add_argument("--use_teacache", action="store_true")
    ra.add_argument(
        "--resample_output_to_source",
        action="store_true",
        help="After infer, resample 81 result frames back to original source count",
    )
    ra.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep resampled fg/mask MP4s in work_dir",
    )
    ra.add_argument("--text_encoder_path", type=str, default=str(d_t5))
    ra.add_argument("--vae_path", type=str, default=str(d_vae))
    ra.add_argument("--dit_path", type=str, default=str(d_dit))
    ra.add_argument("--image_encoder_path", type=str, default=str(d_clip))
    ra.add_argument("--pretrained_lora_path", type=str, default=str(d_lora))
    ra.set_defaults(func=cmd_run_all)
    return p


def main():
    parser = build_parser()
    args, rest = parser.parse_known_args()
    if args.command == "run-all":
        args.infer_extra = rest
        if not args.work_dir:
            args.work_dir = str(
                Path(__file__).resolve().parents[1] / ".effecterase_frame_work"
            )
        if not args.output_video:
            args.output_video = str(Path(args.work_dir) / "remove_out.mp4")
    elif rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    args.func(args)


if __name__ == "__main__":
    main()
