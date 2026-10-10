# Infrastructure

## Recommended pins (verify before a run)

| component | role | note |
|---|---|---|
| PyTorch | tensor + SDPA | 2.4+ for FlashAttention-3 dispatch on Hopper |
| FlashAttention-3 | fused attention | `scripts/install_cluster.sh`, needs SM90 |
| Megatron-LM | TP / PP / EP / SP kernels | pin a commit, do not track main |
| DeepSpeed | ZeRO-1/2/3, MoE | pin a commit |
| vLLM / TensorRT-LLM | serving | fp8/int4 quantized inference |
| WandB | run tracking | loss curves, MFU, expert load |
| Prometheus + Grafana | cluster health | node/cable thermals, ECC, IB errors |

This repo deliberately does not vendor Megatron or DeepSpeed: their exact
revision changes numerics, so it must be pinned in the run's own lockfile.

## Pre-flight checklist

Run before *every* job submission:

```bash
# 1. arithmetic
mega-info --config configs/moe-1t.yaml --gpus 2048 --gpu-mem 141 --tokens 20e12

# 2. topology validation (exits non-zero on mismatch)
python -m mega.train.distributed --config configs/moe-1t.yaml --dry-run

# 3. correctness
python -m pytest tests/ -q
```

The submitted SLURM job (`scripts/slurm/train_moe_1t.sbatch`) also re-validates
the topology inside `distributed.py`, so a config/launch mismatch fails fast
instead of silently training on the wrong number of ranks.

## Cluster sizing notes

**Memory.** Optimizer state dominates. For `P` parameters the static footprint
is `16 * P` bytes (bf16 weights + bf16 grads + fp32 master + 2 fp32 moments).
Shard it: ZeRO-3 over the whole fleet gives
`16 P / n_gpus` bytes per GPU. For the 200 T target that is 3.2 PB / 100 k GPUs
= 32 GB/GPU, which fits — the cost is all-gather bandwidth per step.

**Fabric.** Tensor parallelism must stay inside one NVLink island (8 GPUs);
crossing nodes for TP is a throughput disaster. `plan_parallelism` enforces
`tp <= nvlink_domain`. Expert parallelism and pipeline parallelism can cross
nodes; ensure a rail-optimised IB topology for the all-to-all.

**Storage.** Checkpoints of a large MoE are model-shard-sized at minimum; write
them asynchronously and keep at least two generations. Plan PB-scale NVMe for
the optimizer states and object storage for checkpoints + data shards.

**Energy.** Frontier clusters are 70-200 MW for 100 k GPUs, not 500 MW. Size the
facility to the monitored power draw, and monitor per-node thermals for the
throttling that silently halves MFU.

## MFU expectations

MoE training is all-to-all bound. For a large expert-parallel run, assume
**15-25 % MFU**, not the 40-50 % people quote for dense models. The feasibility
calculator uses 20 % for this reason; changing it moves the wall-time estimate
linearly.

## Failure handling

* Checkpoint every N steps with atomic rename (`utils/checkpoint.py` already
  does `os.replace` so a crash mid-write cannot corrupt the previous file).
* Persist the RNG state with the checkpoint — resume with a different seed
  stream will silently change the loss curve.
* Make the job idempotent: on restart, load the latest valid checkpoint, restore
  RNG, continue. `Trainer.resume` does this.
* Alert on: loss NaN, grad-norm spikes, expert-load skew > 3×, MFU drop > 30 %,
  and node loss events.
