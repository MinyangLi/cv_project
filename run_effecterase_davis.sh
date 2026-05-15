#!/usr/bin/env bash
# Run EffectErase only on DAVIS video pairs:
#   DAVIS/videos/images/*.mp4
#   DAVIS/videos/masks/*.mp4
#
# Output:
#   DAVIS/predicted/effecterase/*.mp4
#
# Usage:
#   bash /hpc2hdd/home/mli861/cv_project/run_effecterase_davis.sh
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 \
#   DAVIS_ROOT=/hpc2hdd/home/mli861/cv_project/DAVIS \
#   WAN_MODEL_ROOT=Wan-AI/Wan2.1-Fun-1.3B-InP \
#   LORA_P=/hpc2hdd/home/mli861/cv_project/EffectErase/EffectErase.ckpt \
#   bash /hpc2hdd/home/mli861/cv_project/run_effecterase_davis.sh

set -euo pipefail

ROOT="/hpc2hdd/home/mli861/cv_project"
EE_DIR="${ROOT}/EffectErase"

DAVIS_ROOT="${DAVIS_ROOT:-${ROOT}/DAVIS}"
DAVIS_IMG_VID_DIR="${DAVIS_ROOT}/videos/images"
DAVIS_MASK_VID_DIR="${DAVIS_ROOT}/videos/masks"
DAVIS_OUT_DIR="${DAVIS_OUT_DIR:-${DAVIS_ROOT}/predicted/effecterase}"

# EffectErase checkpoints (same defaults as inference_all.sh)
WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-Wan-AI/Wan2.1-Fun-1.3B-InP}"
TEXT_ENC="${TEXT_ENC:-${WAN_MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
VAE_P="${VAE_P:-${WAN_MODEL_ROOT}/Wan2.1_VAE.pth}"
DIT_P="${DIT_P:-${WAN_MODEL_ROOT}/diffusion_pytorch_model.safetensors}"
IMG_ENC="${IMG_ENC:-${WAN_MODEL_ROOT}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
LORA_P="${LORA_P:-${EE_DIR}/EffectErase.ckpt}"

sorted_mp4s_in_dir() {
  local dir="$1"
  shopt -s nullglob
  local f
  for f in "$dir"/*.mp4; do
    [[ -f "$f" ]] || continue
    printf '%s\n' "$f"
  done | sort
  shopt -u nullglob
}

if [[ ! -d "$DAVIS_IMG_VID_DIR" || ! -d "$DAVIS_MASK_VID_DIR" ]]; then
  echo "[ERROR] Missing DAVIS dirs:"
  echo "  images: $DAVIS_IMG_VID_DIR"
  echo "  masks : $DAVIS_MASK_VID_DIR"
  echo "Generate videos first if needed:"
  echo "  python ${ROOT}/frames2video.py --root ${DAVIS_ROOT} --overwrite"
  exit 1
fi

mkdir -p "$DAVIS_OUT_DIR"

mapfile -t MP4S < <(sorted_mp4s_in_dir "$DAVIS_IMG_VID_DIR")
if [[ ${#MP4S[@]} -eq 0 ]]; then
  echo "[ERROR] No .mp4 found in $DAVIS_IMG_VID_DIR"
  exit 1
fi

# conda hook can reference unset vars, so disable nounset around hook.
set +u
eval "$(conda shell.bash hook)"
conda activate effecterase
set -u

echo "=============================================================="
echo "EffectErase on DAVIS"
echo "images : $DAVIS_IMG_VID_DIR"
echo "masks  : $DAVIS_MASK_VID_DIR"
echo "output : $DAVIS_OUT_DIR"
echo "model  : $WAN_MODEL_ROOT"
echo "lora   : $LORA_P"
echo "=============================================================="

TOTAL_SEC=0
TOTAL_VIDEOS=0
WALL_START=$(date +%s)

for f in "${MP4S[@]}"; do
  base=$(basename "$f")
  mask_p="${DAVIS_MASK_VID_DIR}/${base}"
  out_p="${DAVIS_OUT_DIR}/${base}"

  if [[ ! -f "$mask_p" ]]; then
    echo "[WARN] Missing mask video for ${base}: ${mask_p} (skip)"
    continue
  fi

  echo "========== EffectErase [DAVIS] ${base} =========="
  t0=$(date +%s)
  (
    cd "$EE_DIR"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python examples/remove_wan/infer_remove_wan.py \
        --fg_bg_path "$f" \
        --mask_path "$mask_p" \
        --output_path "$out_p" \
        --text_encoder_path "$TEXT_ENC" \
        --vae_path "$VAE_P" \
        --dit_path "$DIT_P" \
        --image_encoder_path "$IMG_ENC" \
        --pretrained_lora_path "$LORA_P"
  )
  t1=$(date +%s)
  dt=$((t1 - t0))
  TOTAL_SEC=$((TOTAL_SEC + dt))
  TOTAL_VIDEOS=$((TOTAL_VIDEOS + 1))
  echo "[TIME] ${base} ${dt}s"
done

WALL_END=$(date +%s)
WALL_SEC=$((WALL_END - WALL_START))

echo ""
echo "=============================================================="
echo "EffectErase DAVIS Summary"
echo "videos processed: ${TOTAL_VIDEOS}"
echo "total infer sec : ${TOTAL_SEC}"
if [[ "${TOTAL_VIDEOS}" -gt 0 ]]; then
  awk -v t="$TOTAL_SEC" -v c="$TOTAL_VIDEOS" 'BEGIN { printf "avg/video      : %.2fs\n", t / c }'
fi
echo "wall-clock sec : ${WALL_SEC}"
echo "output dir     : ${DAVIS_OUT_DIR}"
echo "=============================================================="
