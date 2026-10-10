"""Transformer block and the full decoder-only model.

The block is a standard pre-norm residual stack.  A layer uses MoE in place of
the dense FFN when ``(layer_index + 1) % moe_every == 0``; mixing dense and MoE
layers keeps the first layers cheap and the deep layers sparse, which is what
recent large MoE models do.

The model owns the RoPE cache as a *non-persistent* buffer: it is rebuilt from
the config on load, so checkpoints stay small and context extension is a
config change rather than a conversion step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .attention import KVCache, build_attention
from .ffn import SwiGLU
from .moe import MoELayer, MoEStats
from .norms import RMSNorm
from .rope import build_rope_cache

__all__ = ["Block", "MegaModel", "ModelOutput"]


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = build_attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.is_moe = layer_idx in cfg.moe_layer_indices
        if self.is_moe:
            self.ffn: nn.Module = MoELayer(cfg)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, cos, sin, cache: KVCache | None = None, causal: bool = True):
        h, cache = self.attn(self.attn_norm(x), cos, sin, cache=cache, causal=causal)
        x = x + h
        stats = None
        if self.is_moe:
            out, stats = self.ffn(self.ffn_norm(x))  # type: ignore[union-attr]
            x = x + out
        else:
            x = x + self.ffn(self.ffn_norm(x))
        return x, cache, stats


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    stats: list[MoEStats] = field(default_factory=list)
    caches: list[KVCache] | None = None


class MegaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(
            cfg.head_dim if cfg.attn_impl == "gqa" else cfg.qk_rope_head_dim,
            cfg.max_seq_len,
            base=cfg.rope_base,
            scaling=cfg.rope_scaling,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        past_len: int = 0,
        causal: bool = True,
    ) -> ModelOutput:
        b, t = input_ids.shape
        x = self.embed(input_ids)

        cos = self.rope_cos[past_len : past_len + t]
        sin = self.rope_sin[past_len : past_len + t]

        new_caches: list[KVCache] = []
        all_stats: list[MoEStats] = []
        for i, block in enumerate(self.blocks):
            c = None if caches is None else caches[i]
            x, c, stats = block(x, cos, sin, cache=c, causal=causal)
            new_caches.append(c)
            if stats is not None:
                all_stats.append(stats)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        aux = None
        z = None
        if all_stats:
            aux = torch.stack([s.aux_loss for s in all_stats]).mean()
            z = torch.stack([s.z_loss for s in all_stats]).mean()
        return ModelOutput(
            logits=logits,
            loss=loss,
            aux_loss=aux,
            z_loss=z,
            stats=all_stats,
            caches=new_caches if caches is not None else None,
        )

    # ----------------------------------------------------------------- generate
    def empty_caches(self) -> list[KVCache]:
        """One empty cache per layer; pass these to start cached decoding."""
        return [KVCache.empty() for _ in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        caches: list[KVCache] | None = self.empty_caches()
        past = 0
        generated = input_ids
        cur = input_ids

        for _ in range(max_new_tokens):
            out = self.forward(cur, caches=caches, past_len=past, causal=True)
            caches = out.caches
            logits = out.logits[:, -1, :]
            past += cur.shape[1]

            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = logits.softmax(dim=-1)
                if top_p is not None:
                    sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
                    cum = sorted_p.cumsum(dim=-1)
                    keep = cum - sorted_p < top_p
                    sorted_p = sorted_p * keep
                    sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
                    nxt = sorted_idx.gather(
                        -1, torch.multinomial(sorted_p, 1)
                    )
                else:
                    nxt = torch.multinomial(probs, 1)
            generated = torch.cat([generated, nxt], dim=1)
            cur = nxt
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return generated

    # ------------------------------------------------------------- bookkeeping
    def num_parameters(self, non_embedding: bool = False, active: bool = False) -> int:
        """Count parameters.

        ``active=True`` returns the number of parameters a single token touches:
        routed expert matrices are replaced by ``top_k`` expert slices.
        """
        total = 0
        for name, p in self.named_parameters():
            if non_embedding and ("embed." in name or "lm_head." in name):
                continue
            n = p.numel()
            if active and any(k in name for k in (".w_gate", ".w_up", ".w_down")):
                # keep only the parameters that are actually selected
                n = self._active_expert_slice(p, name)
            total += n
        return total

    def _active_expert_slice(self, p: torch.Tensor, name: str) -> int:
        n_experts = p.shape[0]
        per_expert = p.numel() // n_experts
        top_k = min(self.cfg.moe_top_k, n_experts)
        return top_k * per_expert

    def config(self) -> ModelConfig:
        return self.cfg
