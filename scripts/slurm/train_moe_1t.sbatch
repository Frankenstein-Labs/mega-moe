#!/usr/bin/env bash
#SBATCH --job-name=mega-1t
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=gpu-h100
#SBATCH --time=72:00:00
#SBATCH --exclusive
#SBATCH --output=logs/mega-1t-%j.out
#SBATCH --error=logs/mega-1t-%j.err
#
# Launch the 1T MoE training run.
#
# 256 nodes x 8 H100 = 2048 GPUs, matching the parallel plan printed by
#   mega-info --config configs/moe-1t.yaml --gpus 2048
# (tp=8, pp=32, cp=1, dp=8 ; expert parallelism folded into dp).
#
# Before submitting:
#   1. mkdir -p logs
#   2. mega-info --config configs/moe-1t.yaml --gpus 2048 --gpu-mem 141
#   3. verify the plan's tp*pp*cp*dp == 2048
set -euo pipefail

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5          # InfiniBand HCAs
export NCCL_NET_GDR_LEVEL=5
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/moe-1t.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/moe-1t-${SLURM_JOB_ID}}"

# world size / topology must be consistent with the YAML parallel fields
GPUS_PER_NODE=8
NNODES="${SLURM_NNODES:-1}"
WORLD_SIZE=$(( GPUS_PER_NODE * NNODES ))

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=29500

mkdir -p "$OUTPUT_DIR"

echo "world_size=$WORLD_SIZE  master=$MASTER_ADDR:$MASTER_PORT"
echo "config=$CONFIG  out=$OUTPUT_DIR"

# torchrun is the launcher of record; the project's distributed entry point
# (mega.train.distributed) reads the YAML parallel fields and builds the
# process groups.  It is a thin wrapper over the same Trainer used on CPU.
srun --nodes="$NNODES" --ntasks="$WORLD_SIZE" --ntasks-per-node="$GPUS_PER_NODE" \
  torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m mega.train.distributed \
      --config "$CONFIG" \
      --out "$OUTPUT_DIR" \
      --require-backend nccl
