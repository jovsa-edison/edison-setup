#!/bin/bash
# Submit:
#   sbatch train.sbatch                                       # 1 node, 1 GPU
#   sbatch --gpus-per-node=8 --cpus-per-task=8 train.sbatch   # 1 node, 8 GPUs
#   sbatch --nodes=4 --gpus-per-node=8 --cpus-per-task=8 train.sbatch
#
# Pass-through args go to train.py:
#   sbatch train.sbatch --epochs 5 --batch-size 256
#
#SBATCH --job-name=mnist-ddp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%h/logs/%x-%j.out

set -euo pipefail

# Resolve to the directory this sbatch script lives in, so train.py is found
# regardless of where `sbatch` was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$HOME/logs"

# sbatch jobs do not source ~/.bashrc; activate the molr1 venv explicitly.
VENV="${VENV:-$HOME/code/molr1/.venv}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

echo "[sbatch] job=$SLURM_JOB_ID nodes=$SLURM_NNODES gpus-per-node=$GPUS_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py "$@"
