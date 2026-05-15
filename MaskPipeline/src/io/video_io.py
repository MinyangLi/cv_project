from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from src.common.types import OutputPaths, VideoInfo


def inspect_video(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions for {path}")
    return VideoInfo(fps=fps, width=width, height=height, num_frames=num_frames)


def iter_video_frames(path: str | Path, *, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")

    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
            if max_frames is not None and index >= max_frames:
                break
    finally:
        capture.release()


def load_video_frames(path: str | Path, *, max_frames: int | None = None) -> list[np.ndarray]:
    return [frame for _, frame in iter_video_frames(path, max_frames=max_frames)]


class H264VideoWriter:
    def __init__(self, path: str | Path, *, width: int, height: int, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self._writer = imageio_ffmpeg.write_frames(
            str(self.path),
            (width, height),
            pix_fmt_in="rgb24",
            fps=fps,
            codec="libx264",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        self._writer.send(None)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame size mismatch, expected {(self.height, self.width)}, got {frame_bgr.shape[:2]}"
            )
        self._writer.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        self._writer.close()


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    out = (mask > 0).astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise IOError(f"Failed to encode mask png: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def resolve_output_paths(input_video: str | Path, output_video: str | Path | None) -> OutputPaths:
    input_path = Path(input_video)
    if output_video is None:
        if input_path.parent.name == "Unedited" and input_path.parent.parent.name == "Input_video":
            output_mask_video = input_path.parent.parent.parent / "outputs" / f"{input_path.stem}_masked.mp4"
        else:
            output_mask_video = input_path.parent / "outputs" / f"{input_path.stem}_masked.mp4"
    else:
        output_mask_video = Path(output_video)
    stem = output_mask_video.stem.removesuffix("_masked")
    output_root = output_mask_video.parent / stem
    output_overlay_video = output_mask_video.parent / f"{stem}_overlay.mp4"
    masks_dir = output_root / "masks"
    frames_dir = output_root / "frames"
    output_mask_video.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        output_mask_video=output_mask_video,
        output_overlay_video=output_overlay_video,
        output_root=output_root,
        masks_dir=masks_dir,
        frames_dir=frames_dir,
    )
