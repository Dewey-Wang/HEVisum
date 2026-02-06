#!/usr/bin/env bash
#SBATCH -t 1-00:00:00
#SBATCH -J hevisum-2scale
#SBATCH -p earth-5
#SBATCH --gres=gpu:a100:2
#SBATCH --output=/cfs/earth/scratch/wani/Desktop/HEVisum/Slurm-%j/hevisum_%j.out
#SBATCH --error=/cfs/earth/scratch/wani/Desktop/HEVisum/Slurm-%j/hevisum_%j.err


source ~/.bashrc
conda activate HEVision

cd "/cfs/earth/scratch/wani/Desktop/HEVisum"
export PYTHONPATH="/cfs/earth/scratch/wani/Desktop/HEVisum:${PYTHONPATH:-}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || echo "nvidia-smi not found / no driver visible"
# 若你在 login node 解包會因為缺 __cuda，可先加這行（在 GPU node 不需要）
export CONDA_OVERRIDE_CUDA="12.2"

python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
python -u "/cfs/earth/scratch/wani/Desktop/HEVisum/model training/only_tile.py"

python -u "/cfs/earth/scratch/wani/Desktop/HEVisum/model training/2_scale_sub_center.py"
python -u "/cfs/earth/scratch/wani/Desktop/HEVisum/model training/2_scale_tile_sub.py"
python -u "/cfs/earth/scratch/wani/Desktop/HEVisum/model training/2_scale_tile_center.py"
python -u "/cfs/earth/scratch/wani/Desktop/HEVisum/model training/3_scale.py"
