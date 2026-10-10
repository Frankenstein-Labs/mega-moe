"""``mega-info``: print parameter counts, memory, FLOPs and a parallel plan.

This is the tool that answers "does my config fit?".  Run it before every
training job.

Examples
--------
    mega-info --preset small
    mega-info --preset 70b-dense
    mega-info --preset moe-1t --gpus 4096
    mega-info --preset moe-1t --gpus 4096 --tokens 20e12
"""

from __future__ import annotations

import argparse
import json

from ..config import ModelConfig, configs_from_yaml
from ..parallel.plan import (
    active_params,
    kv_cache_bytes,
    plan_parallelism,
    total_params,
    training_flops,
    training_memory,
)

PRESETS: dict[str, ModelConfig] = {
    # tiny: used by the unit tests, trains in seconds on CPU
    "small": ModelConfig(
        vocab_size=4096, max_seq_len=512, n_layers=6, d_model=256, n_heads=8,
        n_kv_heads=2, head_dim=32, ffn_hidden=688, moe_every=3, n_routed_experts=16,
        n_shared_experts=1, moe_top_k=2, init_std=0.02,
    ),
    # a plausible ~70B dense reference point
    "70b-dense": ModelConfig(
        vocab_size=128_000, max_seq_len=8192, n_layers=80, d_model=8192, n_heads=64,
        n_kv_heads=8, head_dim=128, ffn_hidden=28_672, use_moe=False, qk_norm=True,
    ),
    # ~1T-total MoE, ~150B active.  Already at the edge of what is buildable.
    "moe-1t": ModelConfig(
        vocab_size=128_000, max_seq_len=8192, n_layers=61, d_model=7168, n_heads=56,
        n_kv_heads=8, head_dim=128, ffn_hidden=18432, moe_every=1,
        n_routed_experts=48, n_shared_experts=1, moe_top_k=6, router_bias=True,
    ),
    # ~6T-total MoE / ~250B active.  This is the top of the realistically-trained
    # range as of 2025 (DeepSeek-V3 class and slightly beyond).
    "moe-6t": ModelConfig(
        vocab_size=128_000, max_seq_len=8192, n_layers=61, d_model=7168, n_heads=56,
        n_kv_heads=8, head_dim=128, ffn_hidden=18432, moe_every=1,
        n_routed_experts=256, n_shared_experts=2, moe_top_k=8, router_bias=True,
    ),
    # Multi-head latent attention variant of the same shape
    "moe-6t-mla": ModelConfig(
        vocab_size=128_000, max_seq_len=8192, n_layers=61, d_model=7168, n_heads=56,
        n_kv_heads=8, head_dim=128, ffn_hidden=18432, moe_every=1, attn_impl="mla",
        kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, n_routed_experts=256, n_shared_experts=2, moe_top_k=8,
        router_bias=True,
    ),
}

def human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T", "P", "E", "Z", "Y"]:
        if abs(n) < 1000:
            return f"{n:,.2f}{unit}"
        n /= 1000
    return f"{n:,.2f}Y"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MEGA model arithmetic")
    p.add_argument("--preset", choices=sorted(PRESETS), default="small")
    p.add_argument("--config", help="YAML file with model:/train: sections (overrides --preset)")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--gpu-mem", type=float, default=80.0, help="GiB per GPU")
    p.add_argument("--tokens", type=float, default=None, help="training tokens (accepts 20e12)")
    p.add_argument("--seq-len", type=int, default=None, help="override context length")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.config:
        cfg, _ = configs_from_yaml(args.config)
    else:
        cfg = PRESETS[args.preset]
    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    total = total_params(cfg)
    active = active_params(cfg)
    report: dict = {
        "config": {
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "attn_impl": cfg.attn_impl,
            "n_experts": cfg.n_routed_experts if cfg.use_moe else 0,
            "top_k": cfg.moe_top_k if cfg.use_moe else 0,
            "max_seq_len": cfg.max_seq_len,
            "vocab_size": cfg.vocab_size,
        },
        "total_params": total,
        "active_params": active,
        "active_ratio": active / total,
    }

    mem = training_memory(cfg, n_gpus=args.gpus, micro_batch=1)
    report["memory_per_gpu_gib"] = mem

    plan = plan_parallelism(cfg, args.gpus, gpu_mem_gib=args.gpu_mem)
    report["parallel_plan"] = plan.as_dict()

    report["kv_cache_gib_per_1k_seqs"] = kv_cache_bytes(cfg, 1000, cfg.max_seq_len) / 2**30

    if args.tokens:
        flops = training_flops(cfg, int(args.tokens))
        report["flops"] = flops
        # 6N rule sanity: at ~1e15 FLOP/s/GPU (bf16 H100, ~50% MFU)
        gpu_flops = 1e15 * 0.5
        gpu_seconds = flops["training"] / (gpu_flops * args.gpus)
        report["estimated_gpu_days"] = gpu_seconds / 86400

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"=== MEGA :: {args.preset if not args.config else args.config} ===")
    print(f"layers        : {cfg.n_layers}  (MoE layers: {len(cfg.moe_layer_indices)})")
    print(f"d_model       : {cfg.d_model}   attn: {cfg.attn_impl}")
    if cfg.use_moe:
        print(f"experts       : {cfg.n_routed_experts} routed, {cfg.n_shared_experts} shared, top-{cfg.moe_top_k}")
    print(f"context       : {cfg.max_seq_len:,} tokens")
    print(f"vocab         : {cfg.vocab_size:,}")
    print("--")
    print(f"total params  : {human(total)}")
    print(f"active params : {human(active)}  ({active / total * 100:.1f}% of total)")
    print("--")
    print(f"embedding tbl : {human(2 * cfg.vocab_size * cfg.d_model)} (embed + head)")
    print(f"static (ZeRO-0, whole model): {mem['static_per_gpu_gib']:.1f} GiB  (weights+grads+opt)")
    print(f"weights/GPU after TP*PP     : {plan.params_per_gpu_gib:.1f} GiB")
    print(f"activations/GPU             : {mem['activations_gib']:.1f} GiB")
    print("--")
    print(f"parallel plan : {plan.as_dict()}")
    print(f"KV cache      : {report['kv_cache_gib_per_1k_seqs']:.2f} GiB per 1k seqs @ {cfg.max_seq_len:,}")
    if args.tokens:
        print("--")
        print(f"training FLOPs: {human(report['flops']['training'])} "
              f"(attention share {report['flops']['attention_forward'] / report['flops']['forward'] * 100:.0f}%)")
        print(f"est. GPU-days : {report['estimated_gpu_days']:,.1f} on {args.gpus:,} GPUs "
              f"@50% MFU (H100-class)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
