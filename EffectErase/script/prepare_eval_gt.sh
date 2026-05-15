#!/usr/bin/env bash
# Build temporally resampled GT (default 81 frames) for every sequence under eval_dataset/images.
# Same temporal_resample_rgb as EffectErase prepare step — aligns with 81-frame model I/O.
# Run from EffectErase repo root.

set -euo pipefail
cd "$(dirname "$0")/.."

python tools/effecterase_frame_pipeline.py prepare-gt-batch \
  --images_root ../eval_dataset/images \
  --out_root ../eval_dataset/remove_effecterase_gt \
  --num_frames 81
