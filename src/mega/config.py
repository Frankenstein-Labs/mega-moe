"""Configuration objects for the MEGA model family.

Everything that defines a model lives in :class:`ModelConfig` so that a single
object fully determines the parameter count, the memory footprint and the
parallelism plan.  :class:`TrainConfig` carries the optimisation / data
settings.

Both dataclasses round-trip through JSON so checkpoints are self-describing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class ModelConfig:
    # ------------------------------------------------------------------ vocab
    vocab_size: int = 128_000
    max_seq_len: int = 4096

    # ------------------------------------------------------------ transformer
    n_layers: int = 12
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_hidden: int = 2816
    norm_eps: float = 1e-5
    qk_norm: bool = True
    tie_embeddings: bool = False
    dropout: float = 0.0
    init_std: float = 0.02

    # ------------------------------------------------------------- attention
    attn_impl: Literal["gqa", "mla"] = "gqa"
    # Multi-head Latent Attention (DeepSeek-V2/V3 style) dimensions.
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # ----------------------------------------------------------- rope / long
    rope_base: float = 10_000.0
    # e.g. {"type": "yarn", "factor": 8.0, "original_max_seq_len": 4096}
    #      {"type": "linear", "factor": 8.0}
    rope_scaling: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------- MoE
    use_moe: bool = True
    # A layer is a MoE layer when (layer_index + 1) % moe_every == 0.
    moe_every: int = 2
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    moe_top_k: int = 6
    expert_hidden: Optional[int] = None
    router_score: Literal["softmax", "sigmoid"] = "softmax"
    router_jitter: float = 0.0
    # Auxiliary-loss-free balancing (DeepSeek-V3): learned per-expert bias.
    router_bias: bool = False
    aux_loss_coef: float = 1e-3
    z_loss_coef: float = 1e-4

    # -------------------------------------------------------------- computed
    @property
    def moe_layer_indices(self) -> list[int]:
        if not self.use_moe:
            return []
        return [i for i in range(self.n_layers) if (i + 1) % self.moe_every == 0]

    @property
    def resolved_expert_hidden(self) -> int:
        return self.expert_hidden or self.ffn_hidden

    @property
    def q_dim(self) -> int:
        """Per-head query dimension used by the chosen attention impl."""
        if self.attn_impl == "mla":
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.head_dim

    # ------------------------------------------------------------ (de)serial
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> ModelConfig:
        return cls.from_dict(json.loads(text))

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads (GQA)")
        if self.use_moe and self.moe_top_k > self.n_routed_experts:
            raise ValueError("moe_top_k cannot exceed n_routed_experts")
        if self.use_moe and self.n_routed_experts < 1:
            raise ValueError("n_routed_experts must be >= 1")


@dataclass
class TrainConfig:
    # ---- optimisation ----
    optimizer: Literal["muon_adamw", "adamw"] = "muon_adamw"
    lr: float = 3e-4
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 1000
    min_lr_ratio: float = 0.1
    lr_schedule: Literal["cosine", "wsd", "constant"] = "wsd"
    wsd_decay_frac: float = 0.1

    # ---- batching ----
    micro_batch_size: int = 4
    grad_accum_steps: int = 1
    seq_len: int = 512

    # ---- precision ----
    dtype: Literal["fp32", "bf16", "fp16"] = "bf16"
    # FP4/FP8 are only meaningful on Hopper/Blackwell; kept as a target knob.
    quantized_linear: Optional[Literal["fp8", "fp4"]] = None

    # ---- parallelism ----
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    context_parallel: int = 1
    data_parallel: int = 1
    zero_stage: int = 0

    # ---- bookkeeping ----
    seed: int = 0
    log_every: int = 10
    save_every: int = 500
    out_dir: str = "runs/small"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        data = dict(data)
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig fields: {sorted(unknown)}")
        return cls(**data)


def load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def configs_from_yaml(path: str) -> tuple[ModelConfig, TrainConfig]:
    """Load a YAML file containing ``model:`` and ``train:`` sections."""
    raw = load_yaml(path)
    return (
        ModelConfig.from_dict(raw.get("model", {})),
        TrainConfig.from_dict(raw.get("train", {})),
    )
