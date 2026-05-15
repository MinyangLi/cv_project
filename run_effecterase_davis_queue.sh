#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -o log_out/output_effecterase_davis_%j.txt
#SBATCH -e log_out/err_effecterase_davis_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/mli861/cv_project

set -euo pipefail

module load anaconda3
source ~/.bashrc
set +u
eval "$(conda shell.bash hook)"
conda activate effecterase
set -u

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working dir: /hpc2hdd/home/mli861/cv_project"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

cd /hpc2hdd/home/mli861/cv_project

# Optional overrides (uncomment to customize):
# export CUDA_VISIBLE_DEVICES=0
# export DAVIS_ROOT=/hpc2hdd/home/mli861/cv_project/DAVIS
# export DAVIS_OUT_DIR=/hpc2hdd/home/mli861/cv_project/DAVIS/predicted/effecterase
# export WAN_MODEL_ROOT=Wan-AI/Wan2.1-Fun-1.3B-InP
# export LORA_P=/hpc2hdd/home/mli861/cv_project/EffectErase/EffectErase.ckpt

echo "Will run: bash /hpc2hdd/home/mli861/cv_project/run_effecterase_davis.sh"
bash /hpc2hdd/home/mli861/cv_project/run_effecterase_davis.sh

echo "Job ended at $(date)"
