"""Parameter counting, FLOPs, memory and parallelism planning.

These are closed-form estimates, not measurements.  They exist to answer the
only question that matters before writing a training script: *does this fit,
and how long will it take?*  Every formula is written so it can be checked by
hand.

Conventions
-----------
* ``P``   : parameter count
* ``N``   : tokens in a global batch (``micro_batch * grad_accum * seq_len``)
* ``T``   : sequence length
* FLOPs follow the standard ``6 * P * N`` training rule, decomposed so the
  attention term can be separated (it grows with ``T`` and is *not* linear in
  ``P``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from ..config import ModelConfig

__all__ = [
    "total_params",
    "active_params",
    "training_flops",
    "training_memory",
    "kv_cache_bytes",
    "plan_parallelism",
    "ParallelPlan",
]


def _attn_params(cfg: ModelConfig) -> int:
    d = cfg.d_model
    if cfg.attn_impl == "gqa":
        q = d * cfg.n_heads * cfg.head_dim
        kv = 2 * d * cfg.n_kv_heads * cfg.head_dim
        o = d * cfg.n_heads * cfg.head_dim
        norms = 2 * cfg.head_dim if cfg.qk_norm else 0  # q_norm + k_norm
        return q + kv + o + norms
    qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    if cfg.q_lora_rank:
        q = d * cfg.q_lora_rank + cfg.q_lora_rank * cfg.n_heads * qk
        norms = cfg.q_lora_rank  # q_a_norm
    else:
        q = d * cfg.n_heads * qk
        norms = 0
    kv_a = d * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
    kv_b = cfg.kv_lora_rank * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
    o = cfg.n_heads * cfg.v_head_dim * d
    return q + kv_a + kv_b + o + norms + cfg.kv_lora_rank  # + kv_a_norm


def _dense_mlp_params(cfg: ModelConfig) -> int:
    return 3 * cfg.d_model * cfg.ffn_hidden


def _moe_params(cfg: ModelConfig) -> int:
    per = 3 * cfg.d_model * cfg.resolved_expert_hidden
    routed = cfg.n_routed_experts * per
    shared = cfg.n_shared_experts * per
    router = cfg.d_model * cfg.n_routed_experts
    bias = cfg.n_routed_experts if cfg.router_bias else 0
    return routed + shared + router + bias


def total_params(cfg: ModelConfig) -> int:
    """Total parameter count (embeddings included, tied weights counted once)."""
    n = cfg.vocab_size * cfg.d_model  # embedding
    if not cfg.tie_embeddings:
        n += cfg.vocab_size * cfg.d_model  # lm head
    n += cfg.d_model  # final norm
    n_moe = len(cfg.moe_layer_indices)
    n_dense = cfg.n_layers - n_moe
    for _ in range(cfg.n_layers):
        n += _attn_params(cfg) + 2 * cfg.d_model  # norms
    n += n_dense * _dense_mlp_params(cfg)
    n += n_moe * _moe_params(cfg)
    return n


def active_params(cfg: ModelConfig) -> int:
    """Parameters touched by one token, i.e. the cost per forward pass."""
    n = cfg.vocab_size * cfg.d_model  # embedding lookup
    n += cfg.d_model  # final norm
    if not cfg.tie_embeddings:
        n += cfg.vocab_size * cfg.d_model
    n_moe = len(cfg.moe_layer_indices)
    n_dense = cfg.n_layers - n_moe
    for _ in range(cfg.n_layers):
        n += _attn_params(cfg) + 2 * cfg.d_model
    per_expert = 3 * cfg.d_model * cfg.resolved_expert_hidden
    n += n_dense * _dense_mlp_params(cfg)
    n += n_moe * ((cfg.moe_top_k + cfg.n_shared_experts) * per_expert + cfg.d_model * cfg.n_routed_experts)
    return n


def training_flops(cfg: ModelConfig, n_tokens: int, backward_mult: float = 3.0) -> dict[str, float]:
    """Forward FLOPs split into parameter and attention terms.

    Returns a dict with ``forward``, ``training`` (forward + backward) and the
    attention share.
    """
    p_act = active_params(cfg)
    param_flops = 2.0 * p_act * n_tokens

    t = cfg.max_seq_len
    if cfg.attn_impl == "gqa":
        head_cost = cfg.n_heads * cfg.head_dim
    else:
        head_cost = cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
    attn_flops = 4.0 * head_cost * t * n_tokens
    forward = param_flops + attn_flops
    return {
        "param_forward": param_flops,
        "attention_forward": attn_flops,
        "forward": forward,
        "training": backward_mult * forward,
    }


def training_memory(
    cfg: ModelConfig,
    n_gpus: int = 1,
    dtype_bytes: int = 2,
    optimizer: str = "muon_adamw",
    shard_optimizer: bool = False,
    activations_per_layer_bytes: int | None = None,
    micro_batch: int = 1,
) -> dict[str, float]:
    """Per-GPU memory in GiB for a naive data-parallel-ish layout.

    ``optimizer`` states: ``adamw`` uses 2 fp32 moments, ``muon_adamw`` uses a
    bf16 momentum buffer for matrix params and 2 fp32 moments for the rest, so
    it is approximated as 1.5x the parameter count in bf16 plus fp32 master.
    """
    p = total_params(cfg)
    master = 4 * p
    params = dtype_bytes * p
    grads = dtype_bytes * p
    if optimizer == "adamw":
        opt = 8 * p
    else:
        opt = 4 * p + 2 * p  # bf16 momentum for matrices + fp32 moments elsewhere
    static = master + params + grads + opt

    if activations_per_layer_bytes is None:
        # ~ d_model * tokens * a small constant per layer for a few live tensors
        activations_per_layer_bytes = 10 * cfg.d_model * cfg.max_seq_len * micro_batch
    act = activations_per_layer_bytes * cfg.n_layers

    def per_gpu(x: float) -> float:
        return x / n_gpus if shard_optimizer else x

    total = per_gpu(static) + act
    return {
        "params_gib": params / 2**30,
        "grads_gib": grads / 2**30,
        "optimizer_gib": opt / 2**30,
        "static_per_gpu_gib": per_gpu(static) / 2**30,
        "activations_gib": act / 2**30,
        "total_per_gpu_gib": total / 2**30,
    }


def kv_cache_bytes(
    cfg: ModelConfig, batch: int, seq_len: int, dtype_bytes: int = 2, layers: int | None = None
) -> int:
    """KV-cache size in bytes for ``batch`` sequences of ``seq_len`` tokens."""
    layers = layers or cfg.n_layers
    if cfg.attn_impl == "gqa":
        per_token = 2 * cfg.n_kv_heads * cfg.head_dim * dtype_bytes
    else:
        # MLA caches the latent plus the shared rope slice
        per_token = (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * dtype_bytes
    return layers * batch * seq_len * per_token


@dataclass
class ParallelPlan:
    tp: int
    pp: int
    cp: int
    dp: int
    ep: int
    n_gpus: int
    params_per_gpu_gib: float
    layers_per_stage: float
    feasible: bool
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "pp": self.pp,
            "cp": self.cp,
            "dp": self.dp,
            "ep": self.ep,
            "n_gpus": self.n_gpus,
            "params_per_gpu_gib": round(self.params_per_gpu_gib, 2),
            "layers_per_stage": round(self.layers_per_stage, 2),
            "feasible": self.feasible,
            "notes": self.notes,
        }


def plan_parallelism(
    cfg: ModelConfig,
    n_gpus: int,
    gpu_mem_gib: float = 80.0,
    max_tp: int = 8,
    max_pp: int | None = None,
    nvlink_domain: int = 8,
) -> ParallelPlan:
    """Search factorisations of ``n_gpus`` for a workable 3D/4D/5D plan.

    Heuristics, all of which reflect real hardware constraints:

    * ``tp`` never exceeds ``nvlink_domain`` (tensor parallelism is
      communication-heavy and must stay inside one NVLink island).
    * ``tp`` never exceeds ``d_model / 128`` because TP shards the hidden dim
      and every shard must stay a useful matrix.
    * ``pp`` never exceeds the number of layers.
    * expert parallelism ``ep`` must divide the number of routed experts and is
      folded into the data-parallel dimension.

    The score minimises a cost model: static memory pressure plus a
    communication penalty that grows with ``tp`` and shrinks with ``dp``.
    """
    p = total_params(cfg)
    max_pp = max_pp or cfg.n_layers
    best: ParallelPlan | None = None

    for tp, pp, cp in itertools.product(
        [2**i for i in range(0, 8)],
        [2**i for i in range(0, 8)],
        [2**i for i in range(0, 4)],
    ):
        if tp > max_tp or pp > max_pp or cp > n_gpus:
            continue
        if tp > max(1, cfg.d_model // 128):
            continue
        if tp > nvlink_domain:
            continue
        rest = n_gpus // (tp * pp * cp)
        if rest < 1 or tp * pp * cp * rest != n_gpus:
            continue
        dp = rest
        ep = 1
        if cfg.use_moe:
            for cand in sorted((d for d in range(1, dp + 1) if dp % d == 0), reverse=True):
                if cfg.n_routed_experts % cand == 0 and cand <= cfg.n_routed_experts:
                    ep = cand
                    break
        model_shard = p / (tp * pp)
        params_gib = model_shard * 2 / 2**30  # bf16 weights only
        layers_per_stage = cfg.n_layers / pp
        # communication penalty: TP is the most expensive, then PP, then DP
        penalty = 1.0 * math.log2(tp + 1) + 0.5 * math.log2(pp + 1) - 0.3 * math.log2(dp + 1)
        score = params_gib / max(gpu_mem_gib, 1) + 0.02 * penalty
        feasible = params_gib < 0.6 * gpu_mem_gib and layers_per_stage >= 1
        plan = ParallelPlan(
            tp=tp, pp=pp, cp=cp, dp=dp, ep=ep, n_gpus=n_gpus,
            params_per_gpu_gib=params_gib, layers_per_stage=layers_per_stage,
            feasible=feasible,
            notes="" if feasible else "weights alone exceed the memory budget",
        )
        if best is None or score < best_score:  # type: ignore[name-defined]
            best, best_score = plan, score
    if best is None:
        raise ValueError(f"no valid parallel plan for n_gpus={n_gpus}")
    return best
