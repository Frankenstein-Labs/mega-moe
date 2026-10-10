"""Reproducible feasibility report.

The numbers quoted in ``docs/FEASIBILITY.md`` come from here.  Run::

    python -c "from mega.utils.feasibility import report; report()"

Every figure is computed from :mod:`mega.parallel.plan`, so if you change a
formula the document can be regenerated rather than hand-edited.
"""

from __future__ import annotations

from ..config import ModelConfig
from ..parallel.plan import (
    active_params,
    kv_cache_bytes,
    total_params,
    training_flops,
)
from .calc import human

# H100 SXM bf16 dense peak, and the MFU we are willing to assume for MoE runs.
H100_BF16 = 989e12
MOE_MFU = 0.20
DENSE_MFU = 0.40


def _spec_matching_config() -> ModelConfig:
    """A config matching the brief's headline shape."""
    return ModelConfig(
        vocab_size=256_000,
        max_seq_len=10_000_000,
        n_layers=100,
        d_model=8192,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        ffn_hidden=16_384,
        moe_every=1,
        n_routed_experts=40_000,
        n_shared_experts=2,
        moe_top_k=64,
        expert_hidden=2048,
        router_bias=True,
    )


def report(tokens: float = 20e12, gpus: int = 100_000) -> dict:
    cfg = _spec_matching_config()
    p = total_params(cfg)
    a = active_params(cfg)
    flops = training_flops(cfg, int(tokens))["training"]

    per_gpu = H100_BF16 * MOE_MFU
    wall_days = flops / (per_gpu * gpus) / 86400

    mla = ModelConfig.from_dict(cfg.to_dict())
    mla.attn_impl = "mla"

    out = {
        "total_params": p,
        "active_params": a,
        "sparsity_ratio": p / a,
        "training_flops": flops,
        "wall_days_100k_h100_at_20pct_mfu": wall_days,
        "optimizer_state_bytes": 16 * p,
        "fp8_weights_bytes": p,
        "kv_gib_per_seq": {
            f"{ctx}": kv_cache_bytes(mla, 1, ctx) / 2**30
            for ctx in (128_000, 1_000_000, 10_000_000, 20_000_000)
        },
    }

    print("=== MEGA feasibility report ===")
    print(f"total params        : {human(p)}")
    print(f"active params/token : {human(a)}  (ratio {p / a:,.0f}:1)")
    print(f"experts             : {cfg.n_routed_experts:,} routed, top-{cfg.moe_top_k}")
    print("--")
    print(f"training FLOPs @{human(tokens)} tokens : {human(flops)}")
    print(f"  wall time         : {wall_days:,.0f} days on {gpus:,} H100 @ {MOE_MFU:.0%} MFU")
    print(f"  i.e.              : {wall_days / 30:,.1f} months")
    print("--")
    print(f"optimizer state     : {human(16 * p)} bytes (bf16 w/g + fp32 master + 2 moments)")
    print(f"fp8 weights         : {human(p)} bytes")
    print(f"bf16 weights        : {human(2 * p)} bytes")
    print("--")
    print("KV cache per single sequence (MLA):")
    for ctx, gib in out["kv_gib_per_seq"].items():
        print(f"  {int(ctx):>12,} tokens : {gib:>10,.1f} GiB")
    print("--")
    print("Verdict: parameter count is frontier-edge but consistent.")
    print("         Compute is months-scale on a large fleet; optimizer memory is a")
    print("         sharding problem.  The 10M/20M context and 'FP4 with <0.5% loss'")
    print("         requirements are the parts that are not achievable as stated.")
    return out


if __name__ == "__main__":  # pragma: no cover
    report()
