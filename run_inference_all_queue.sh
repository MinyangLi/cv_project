#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -o log_out/output_%j.txt
#SBATCH -e log_out/err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/mli861/cv_project

# 本作业会依次用 rose / effecterase / propainter 三个 conda 环境（见 inference_all.sh）。
# CUDA 模块选集群可用且与三者 wheel 兼容的版本；若无 cuda/12.4，用 module avail cuda 替换。
# module load cuda/12.4
module load anaconda3

source ~/.bashrc

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Will run: bash /hpc2hdd/home/mli861/cv_project/inference_all.sh"

cd /hpc2hdd/home/mli861/cv_project

# 可选：指定 GPU、Wan 权重根目录、EffectErase LoRA 绝对路径
# export CUDA_VISIBLE_DEVICES=0
# export WAN_MODEL_ROOT=/path/to/Wan2.1-Fun-1.3B-InP
# export LORA_P=/hpc2hdd/home/mli861/cv_project/EffectErase/EffectErase.ckpt

bash inference_all.sh

echo "Job ended at $(date)"
