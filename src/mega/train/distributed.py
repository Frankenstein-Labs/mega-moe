"""Distributed entry point.

This is a *supervision* wrapper rather than a full Megatron replacement.  It
does three things honestly:

1. reads ``SLURM_*`` / torchrun variables and validates that the YAML parallel
   plan multiplies to the real world size;
2. initialises the process group and reports per-rank GPU memory;
3. hands off to the single-process :class:`~mega.train.loop.Trainer` on rank 0
   while keeping every rank alive, so the launch path is exercised end to end.

The tensor/pipeline/expert-parallel *compute* kernels live in Megatron-LM and
DeepSpeed; wiring them up is an integration task, not something to fake.  Until
that integration lands, this wrapper makes the failure mode explicit instead of
silently training on one GPU.
"""

from __future__ import annotations

import argparse
import os

import torch

from ..config import configs_from_yaml
from ..data.pack import SyntheticCorpus
from ..parallel.plan import plan_parallelism
from ..train.loop import Trainer

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distributed MEGA entry point")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--require-backend", default="nccl")
    p.add_argument("--dry-run", action="store_true", help="validate the plan and exit")
    return p


def _world_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, world, local_rank = _world_info()
    is_main = rank == 0

    model_cfg, train_cfg = configs_from_yaml(args.config)
    if args.out:
        train_cfg.out_dir = args.out

    # 1. validate the declared topology against reality
    declared = (
        train_cfg.tensor_parallel * train_cfg.pipeline_parallel
        * train_cfg.context_parallel * train_cfg.data_parallel
    )
    # a dry run on a single process just reports; a real launch must match
    enforce = world > 1 and not args.dry_run
    if declared != 1 and enforce and declared != world:
        msg = (
            f"parallel plan in config multiplies to {declared} but world_size is {world}. "
            "Fix the YAML before submitting."
        )
        if is_main:
            print(f"ERROR: {msg}")
        raise SystemExit(2)

    if is_main:
        plan = plan_parallelism(model_cfg, world, gpu_mem_gib=80.0)
        print(f"[rank0] world_size={world}  suggested plan={plan.as_dict()}")

    # 2. initialise the process group when we actually have GPUs
    if torch.cuda.is_available():
        backend = args.require_backend
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world)
        if is_main:
            free, total = torch.cuda.mem_get_info()
            print(f"[rank0] {torch.cuda.get_device_name()}  "
                  f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
    elif is_main:
        print("[rank0] CUDA unavailable -- running the reference loop on CPU. "
              "Real tensor/pipeline/expert parallelism requires the Megatron/DeepSpeed "
              "integration described in docs/INFRASTRUCTURE.md.")

    if args.dry_run:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return 0

    # 3. hand off to the reference trainer (rank 0 runs it; other ranks hold)
    if is_main:
        corpus = SyntheticCorpus(
            model_cfg.vocab_size, train_cfg.seq_len, n_samples=512, seed=train_cfg.seed
        )
        Trainer(model_cfg, train_cfg).train(corpus)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
