# MaskPipeline

`cv_project/MaskPipeline` is a video-level object mask pipeline that prioritizes stable whole-video masks instead of frame-by-frame independent segmentation.

## What It Does

Input:

- `Input_video/Unedited/*.mp4`
- or any explicit `--input` path

Output:

- `outputs/<stem>_masked.mp4`
- `outputs/<stem>_overlay.mp4`
- `outputs/<stem>/masks/*.png`
- `outputs/<stem>/meta.json`

## Recommended CLI

Single-instance tracking with a box prompt:

```bash
python scripts/run_mask_video.py \
  --input "Input_video/Unedited/example-3.mp4" \
  --target-text "the man in white shirt" \
  --yolo-class person \
  --prompt-box "100,120,300,500" \
  --output outputs/example-3_masked.mp4
```

Text-guided auto-instance mode:

```bash
python scripts/run_mask_video.py \
  --input "Input_video/Unedited/example-3.mp4" \
  --target-text "person" \
  --yolo-class person \
  --output outputs/example-3_masked.mp4
```

Deprecated compatibility mode:

```bash
python scripts/run_mask_video.py \
  --input "Input_video/Unedited/example-3.mp4" \
  --target-class person \
  --output outputs/example-3_masked.mp4
```

## Instance Semantics

- If `--prompt-box` is provided:
  the pipeline treats it as the recommended single-instance mode and tracks that specific object.
- If `--prompt-box` is not provided:
  the system auto-selects one instance. This may not be the object the user intended.
- Only explicit `--all-instances` returns the union of all matching instances.

## Main Pipeline

1. Use SAM3 tracker video for a specific instance if `--prompt-box` is given.
2. Otherwise use SAM3 text-video segmentation to find the concept specified by `--target-text`.
3. If auto-instance mode is active, select one instance and switch back to tracker mode for stable propagation.
4. If SAM3 masks are weak or missing, use YOLOv8-seg single-instance fallback.
5. Optionally refine masks with optical flow.
6. Save binary masks, an overlay video, and `meta.json`.

## Install

Use the `sam3` conda environment:

```bash
source /home/czy/anaconda3_2/etc/profile.d/conda.sh
conda activate sam3
```

Current environment facts observed during implementation:

- `opencv-python`, `numpy`, `PyYAML`, `imageio-ffmpeg` are already installed
- `torch`, `torchvision`, `ultralytics`, `transformers`, `accelerate` are missing
- local SAM3 model directory exists at:
  `/home/czy/.cache/modelscope/hub/models/facebook/sam3`

Install the missing packages:

```bash
pip install torch torchvision
pip install ultralytics transformers accelerate
```

If your `transformers` build does not provide `Sam3VideoModel`, `Sam3VideoProcessor`, `Sam3TrackerVideoModel`, and `Sam3TrackerVideoProcessor`, upgrade to a version that includes SAM3 support.

## If SAM3 Is Missing Or Unavailable

There are two separate requirements:

1. the SAM3 model directory
2. Python code support via `transformers` Sam3 classes

This project already points at the local model directory:

`/home/czy/.cache/modelscope/hub/models/facebook/sam3`

If the model directory exists but the environment cannot import the SAM3 classes, the code will fail with a clear error message.

For debugging the rest of the pipeline without SAM3, you can use:

```bash
python scripts/run_mask_video.py \
  --input "Input_video/Unedited/example-3.mp4" \
  --target-text "person" \
  --yolo-class person \
  --allow-yolo-only-fallback \
  --max-frames 30 \
  --save-frames \
  --save-debug
```

This requires `ultralytics`, but not working SAM3 support.

## Debug Flags

- `--max-frames`
- `--save-frames`
- `--save-debug`
- `--enable-optical-flow`
- `--disable-optical-flow`

These are useful for short test runs before processing a full video.

## Project Structure

- `scripts/run_mask_video.py`: CLI entrypoint
- `src/cli.py`: argument parsing and config assembly
- `src/pipeline/video_mask_pipeline.py`: main orchestration
- `src/segmentation/sam3_adapter.py`: SAM3 loading and inference adapter
- `src/segmentation/yolo_seg_fallback.py`: YOLOv8-seg fallback and single-instance selection
- `src/detection/yolo_prompt_detector.py`: YOLO detection utilities
- `src/refine/optical_flow_refine.py`: optional optical-flow refinement
- `src/postprocess/mask_postprocess.py`: binary mask cleanup
- `src/io/video_io.py`: video reading and H.264 writing
- `src/io/debug_io.py`: overlay rendering, png output, meta output
- `configs/mask_video.yaml`: default settings

## Minimal Test Commands

Syntax and imports:

```bash
python -m py_compile scripts src
```

CLI help:

```bash
python scripts/run_mask_video.py --help
```

Short fallback-only smoke test:

```bash
python scripts/run_mask_video.py \
  --input "Input_video/Unedited/example-3.mp4" \
  --target-text "person" \
  --yolo-class person \
  --allow-yolo-only-fallback \
  --max-frames 30 \
  --output outputs/example-3_masked.mp4
```
