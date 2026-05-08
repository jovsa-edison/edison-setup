#!/bin/bash
# Submit (template: sbatch [SLURM_OPTS] train.sbatch [TRAIN_ARGS]):
#
# Single node:
#   sbatch train.sbatch                                                  # 1 node x 1 GPU  (default; 1 rank)
#   sbatch --gpus-per-node=8 --cpus-per-task=8 train.sbatch              # 1 node x 8 GPUs (8 ranks)
#
# Multi-node DDP (just bump --nodes=N; everything else stays the same):
#   sbatch --nodes=2 --gpus-per-node=8 --cpus-per-task=8 train.sbatch    # 2 nodes x 8 GPUs (16 ranks)
#   sbatch --nodes=4 --gpus-per-node=8 --cpus-per-task=8 train.sbatch    # 4 nodes x 8 GPUs (32 ranks)
#   sbatch --nodes=8 --gpus-per-node=8 --cpus-per-task=8 train.sbatch    # 8 nodes x 8 GPUs (64 ranks)
#
# Pass-through args go to train.py (after the .sbatch path):
#   sbatch train.sbatch --epochs 5 --batch-size 256
#   sbatch --nodes=4 --gpus-per-node=8 --cpus-per-task=8 train.sbatch --epochs 10
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

# sbatch jobs do not source ~/.bashrc; ensure uv is on PATH.
export PATH="$HOME/.local/bin:$PATH"

# Materialize .venv from uv.lock once on the login pod. No-op if already in
# sync. --locked fails loudly if pyproject.toml and uv.lock have drifted.
uv sync --locked

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

echo "[sbatch] job=$SLURM_JOB_ID nodes=$SLURM_NNODES gpus-per-node=$GPUS_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

# --no-sync skips the venv freshness check inside each Slurm task; we already
# synced above, and the .venv lives on shared NFS so all workers see it.
srun --kill-on-bad-exit=1 uv run --no-sync torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py "$@"
