#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -o log_out_concise/eval_output_%j.txt
#SBATCH -e log_out_concise/eval_err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/mli861/cv_project

# 全 Benchmark 评测（PSNR / SSIM / LPIPS(vgg) / VFID）
# 使用脚本: eval_all.py（自定义实现，VFID 仅借助 I3D 预训练权重）
# 提交方式: sbatch run_eval_all_queue.sh

module load cuda/12.4
module load anaconda3

source ~/.bashrc

ROOT="/hpc2hdd/home/mli861/cv_project"
BENCH="${ROOT}/ROSE-Benchmark/Benchmark"
EVAL_FRAMES="${EVAL_FRAMES:-81}"
I3D="${I3D:-${ROOT}/ProPainter/weights/i3d_rgb_imagenet.pt}"
OUT_JSON="${OUT_JSON:-${ROOT}/eval_results/eval_all_results_${SLURM_JOB_ID:-manual}.json}"
OUT_CSV="${OUT_CSV:-${ROOT}/eval_results/eval_all_results_${SLURM_JOB_ID:-manual}.csv}"

mkdir -p "${ROOT}/log_out"
cd "${ROOT}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
LOG_ID="${SLURM_JOB_ID:-manual}"
echo "BENCH=${BENCH}  EVAL_FRAMES=${EVAL_FRAMES}  LOG_ID=${LOG_ID}"
echo "I3D=${I3D}"
echo "OUT_JSON=${OUT_JSON}"
echo "OUT_CSV=${OUT_CSV}"

set +u
eval "$(conda shell.bash hook)"
conda activate propainter
set -u

python eval_all.py \
  --benchmark_root "${BENCH}" \
  --methods effecterase propainter rose \
  --max_frames "${EVAL_FRAMES}" \
  --i3d_model_path "${I3D}" \
  --out_json "${OUT_JSON}" \
  --existing_json /hpc2hdd/home/mli861/cv_project/eval_results/eval_all_results_9683195.json \
  --out_csv "${OUT_CSV}" \
  --update_mse_mae_only

echo "Job ended at $(date)"
