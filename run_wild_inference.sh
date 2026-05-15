#!/usr/bin/env bash
set -euo pipefail

ROOT="/hpc2hdd/home/mli861/cv_project"
OUT="${OUT:-${ROOT}/wild_videos/inference_outputs}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-864}"
NUM_FRAMES="${NUM_FRAMES:-81}"

ROSE_DIR="${ROOT}/ROSE"
EE_DIR="${ROOT}/EffectErase"
PP_DIR="${ROOT}/ProPainter"

TEXT_ENC="${EE_DIR}/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth"
VAE_P="${EE_DIR}/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth"
DIT_P="${EE_DIR}/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"
IMG_ENC="${EE_DIR}/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
LORA_P="${EE_DIR}/EffectErase.ckpt"

NAMES=(
  "wild_people"
)

VIDEOS=(
  "${ROOT}/wild_videos/wild_people.mp4"
)

MASKS=(
  "${ROOT}/wild_videos/wild_people_mask.mp4"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing directory: $path" >&2
    exit 1
  fi
}

require_dir "$ROSE_DIR"
require_dir "$EE_DIR"
require_dir "$PP_DIR"
require_file "$TEXT_ENC"
require_file "$VAE_P"
require_file "$DIT_P"
require_file "$IMG_ENC"
require_file "$LORA_P"

for path in "${VIDEOS[@]}" "${MASKS[@]}"; do
  require_file "$path"
done

mkdir -p "${OUT}/rose" "${OUT}/effecterase" "${OUT}/propainter"

# Conda activation scripts can reference unset variables, so keep nounset off while switching envs.
set +u
eval "$(conda shell.bash hook)"
set -u

echo "================================================================================"
echo "Running wild video inference for wild_people"
echo "OUT:   $OUT"
echo "GPU:   $GPU"
echo "SIZE:  ${WIDTH}x${HEIGHT}"
echo "================================================================================"

run_rose() {
  local name="$1"
  local video="$2"
  local mask="$3"

  echo "========== ROSE: ${name} (${WIDTH}x${HEIGHT}) =========="
  (
    set +u
    cd "$ROSE_DIR"
    conda activate rose
    CUDA_VISIBLE_DEVICES="$GPU" python inference.py \
      --validation_videos "$video" \
      --validation_masks "$mask" \
      --validation_prompts "" \
      --output_dir "${OUT}/rose" \
      --video_length "$NUM_FRAMES" \
      --sample_size "$HEIGHT" "$WIDTH"
  )
}

run_effecterase() {
  local name="$1"
  local video="$2"
  local mask="$3"

  echo "========== EffectErase: ${name} (${WIDTH}x${HEIGHT}) =========="
  (
    set +u
    cd "$EE_DIR"
    conda activate effecterase
    CUDA_VISIBLE_DEVICES="$GPU" python examples/remove_wan/infer_remove_wan.py \
      --fg_bg_path "$video" \
      --mask_path "$mask" \
      --output_path "${OUT}/effecterase/${name}.mp4" \
      --num_frames "$NUM_FRAMES" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --text_encoder_path "$TEXT_ENC" \
      --vae_path "$VAE_P" \
      --dit_path "$DIT_P" \
      --image_encoder_path "$IMG_ENC" \
      --pretrained_lora_path "$LORA_P"
  )
}

run_propainter() {
  local name="$1"
  local video="$2"
  local mask="$3"

  echo "========== ProPainter: ${name} (${WIDTH}x${HEIGHT}) =========="
  (
    set +u
    cd "$PP_DIR"
    conda activate propainter
    CUDA_VISIBLE_DEVICES="$GPU" python inference_propainter.py \
      --video "$video" \
      --mask "$mask" \
      --output "${OUT}/propainter" \
      --height "$HEIGHT" \
      --width "$WIDTH"
  )
}

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  video="${VIDEOS[$i]}"
  mask="${MASKS[$i]}"

  echo "################################################################################"
  echo "Input: ${name}"
  echo "VIDEO: ${video}"
  echo "MASK:  ${mask}"
  echo "SIZE:  ${WIDTH}x${HEIGHT}"
  echo "################################################################################"

  run_rose "$name" "$video" "$mask"
  run_effecterase "$name" "$video" "$mask"
  run_propainter "$name" "$video" "$mask"
done

echo "================================================================================"
echo "[DONE] Outputs:"
for name in "${NAMES[@]}"; do
  echo "  ${name}"
  echo "    ROSE:        ${OUT}/rose/${name}.mp4"
  echo "    EffectErase: ${OUT}/effecterase/${name}.mp4"
  echo "    ProPainter:  ${OUT}/propainter/${name}.mp4"
done
echo "================================================================================"
