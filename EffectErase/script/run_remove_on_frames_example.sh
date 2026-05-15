#!/usr/bin/env bash
# Batch: sequences under eval_dataset/images -> EffectErase remove -> eval_dataset/remove_effecterase/<seq>/
# Run from EffectErase repo root.
#
# Optional env:
#   RESAMPLE_TO_SOURCE=1
#       append --resample_output_to_source (output frame count = original, not 81).
#   RUN_SLICE_START / RUN_SLICE_END  (1-based inclusive, after sorting names)
#       e.g. only 8th–14th videos (by sorted folder name):
#         RUN_SLICE_START=8 RUN_SLICE_END=14 ./script/run_remove_on_frames_example.sh
#       If both unset, process all sequences. If only START set, runs START..last.
#       If only END set, runs 1..END.

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES_ROOT="../eval_dataset/images"
MASKS_ROOT="../eval_dataset/test_masks"
OUT_ROOT="../eval_dataset/remove_effecterase"

EXTRA_ARGS=()
if [[ "${RESAMPLE_TO_SOURCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resample_output_to_source)
fi

shopt -s nullglob
mapfile -t ALL_NAMES < <(
  for d in "$IMAGES_ROOT"/*/; do
    [[ -d "$d" ]] || continue
    basename "$d"
  done | sort
)

n=${#ALL_NAMES[@]}
if (( n == 0 )); then
  echo "No sequence folders under $IMAGES_ROOT"
  exit 1
fi

if [[ -z "${RUN_SLICE_START:-}" && -z "${RUN_SLICE_END:-}" ]]; then
  SELECTED=("${ALL_NAMES[@]}")
  echo "[INFO] Running all $n sequences (sorted by name)"
else
  start="${RUN_SLICE_START:-1}"
  end="${RUN_SLICE_END:-$n}"
  if (( start < 1 || end < 1 || start > n || end > n || start > end )); then
    echo "[ERROR] Invalid slice: RUN_SLICE_START=$start RUN_SLICE_END=$end (have $n sequences, indices 1..$n)"
    exit 1
  fi
  offset=$((start - 1))
  len=$((end - start + 1))
  SELECTED=("${ALL_NAMES[@]:offset:len}")
  echo "[INFO] Slice $start-$end of $n sequences (sorted): ${SELECTED[*]}"
fi

for name in "${SELECTED[@]}"; do
  seq_dir="${IMAGES_ROOT}/${name}/"
  mask_dir="${MASKS_ROOT}/${name}"
  out_dir="${OUT_ROOT}/${name}"

  if [[ ! -d "$mask_dir" ]]; then
    echo "[SKIP] no mask dir: $mask_dir"
    continue
  fi

  echo "========== Processing: $name =========="
  python tools/effecterase_frame_pipeline.py run-all \
    --rgb_dir "$seq_dir" \
    --mask_dir "$mask_dir" \
    --output_frames_dir "$out_dir" \
    "${EXTRA_ARGS[@]}" \
    --num_frames 81 \
    --duration_sec 3.0 \
    --text_encoder_path Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth \
    --dit_path Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors \
    --image_encoder_path Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --pretrained_lora_path EffectErase.ckpt
done

echo "[DONE] processed ${#SELECTED[@]} sequence(s) under $IMAGES_ROOT"
