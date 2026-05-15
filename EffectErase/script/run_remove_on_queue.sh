#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -o log_out/output_%j.txt
#SBATCH -e log_out/err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/mli861/cv_project/EffectErase

# 与 effecterase 环境 PyTorch 的 CUDA 12.8 对齐；若集群无 cuda/12.8，用 module avail cuda 选接近版本
module load cuda/12.8
module load anaconda3

source ~/.bashrc
conda activate effecterase

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Using conda environment: $CONDA_DEFAULT_ENV"
echo "PyTorch / CUDA:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" || true
command -v nvcc >/dev/null && nvcc --version || echo "nvcc not in PATH (often OK if only using PyTorch wheels)"

cd /hpc2hdd/home/mli861/cv_project/EffectErase

# 可选（与 run_remove_on_frames_example.sh 一致）：
#   export WAN_MODEL_ROOT=/path/to/Wan2.1-Fun-1.3B-InP   # 含四个权重文件的目录
#   export EFFECTERASE_CKPT=/path/to/EffectErase.ckpt
export RESAMPLE_TO_SOURCE=1
export RUN_SLICE_START=1
export RUN_SLICE_END=2
bash script/run_remove_on_frames_example.sh

echo "Job ended at $(date)"
