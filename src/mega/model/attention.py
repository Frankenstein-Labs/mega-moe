"""Attention: grouped-query attention (GQA) and multi-head latent attention (MLA).

Both variants share the same forward signature and the same KV-cache contract,
so the transformer can swap them without any other change::

    out, new_cache = attn(x, cos, sin, cache=prev_cache)

* **GQA** (Ainslie et al., 2023): ``n_heads`` query heads share ``n_kv_heads``
  key/value heads, cutting the KV cache by ``n_heads / n_kv_heads``.
* **MLA** (DeepSeek-V2/V3): keys and values are produced from a single low-rank
  latent ``c_kv`` of width ``kv_lora_rank``; a small ``qk_rope_head_dim`` slice
  carries position.  The cache holds ``(c_kv, k_rope)`` only, which is far
  smaller than a full multi-head cache.

``torch.nn.functional.scaled_dot_product_attention`` is used so the Flash
Attention kernels are picked up automatically on Hopper/Blackwell.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm
from .rope import apply_rope

__all__ = ["KVCache", "GQAttention", "MLAttention", "build_attention"]


@dataclass
class KVCache:
    """Per-layer key/value cache.  Shape ``[batch, heads, seq, dim]``."""

    k: torch.Tensor
    v: torch.Tensor

    @property
    def seq_len(self) -> int:
        return 0 if self.k.numel() == 0 else self.k.shape[-2]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> KVCache:
        if self.k.numel() == 0:
            return KVCache(k, v)
        return KVCache(torch.cat([self.k, k], dim=-2), torch.cat([self.v, v], dim=-2))

    @classmethod
    def empty(cls) -> KVCache:
        return cls(torch.empty(0), torch.empty(0))


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class _AttentionBase(nn.Module):
    n_rep: int = 1

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _supports_gqa_kernel(self) -> bool:
        if not hasattr(self, "_gqa_kernel"):
            try:
                q = torch.zeros(1, 2, 1, 4)
                kv = torch.zeros(1, 1, 1, 4)
                F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
                self._gqa_kernel = True
            except TypeError:
                self._gqa_kernel = False
        return self._gqa_kernel

    def _sdpa(self, q, k, v, past: int, causal: bool) -> torch.Tensor:
        t = q.shape[-2]
        s = k.shape[-2]
        attn_mask = None
        is_causal = False
        if causal:
            if past == 0 and t == s:
                is_causal = True
            else:
                qidx = torch.arange(past, past + t, device=q.device).view(t, 1)
                kidx = torch.arange(s, device=q.device).view(1, s)
                attn_mask = torch.zeros(t, s, device=q.device, dtype=q.dtype)
                attn_mask.masked_fill_(kidx > qidx, float("-inf"))
        use_gqa = self.n_rep > 1 and self._supports_gqa_kernel()
        kk = k if use_gqa else _repeat_kv(k, self.n_rep)
        vv = v if use_gqa else _repeat_kv(v, self.n_rep)
        kwargs = dict(attn_mask=attn_mask, is_causal=is_causal, dropout_p=0.0)
        if use_gqa:
            kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(q, kk, vv, **kwargs)


class GQAttention(_AttentionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x, cos, sin, cache: KVCache | None = None, causal: bool = True):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q.transpose(1, 2), cos, sin)
        k = apply_rope(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)

        past = 0
        if cache is not None:
            past = cache.seq_len
            cache = cache.append(k, v)
            k, v = cache.k, cache.v
        out = self._sdpa(q, k, v, past, causal)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), cache


class MLAttention(_AttentionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.qk_dim = self.qk_nope + self.qk_rope
        self.n_rep = 1
        self.q_lora_rank = cfg.q_lora_rank

        if self.q_lora_rank:
            self.q_a_proj = nn.Linear(cfg.d_model, self.q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(self.q_lora_rank, cfg.norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, cfg.n_heads * self.qk_dim, bias=False)
        else:
            self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.qk_dim, bias=False)

        self.kv_a_proj = nn.Linear(cfg.d_model, self.kv_lora_rank + self.qk_rope, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora_rank, cfg.norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, cfg.n_heads * (self.qk_nope + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(cfg.n_heads * self.v_head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None, causal: bool = True):
        b, t, _ = x.shape
        if self.q_lora_rank:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.view(b, t, self.n_heads, self.qk_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        c_kv, k_rope = self.kv_a_proj(x).split([self.kv_lora_rank, self.qk_rope], dim=-1)
        k_rope = k_rope.view(b, t, 1, self.qk_rope).transpose(1, 2)

        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        past = 0
        if cache is not None:
            past = cache.seq_len
            cache = cache.append(c_kv.unsqueeze(1), k_rope)
            c_cached, k_rope = cache.k, cache.v
            c_cached = c_cached.squeeze(1)  # [b, s, kv_lora_rank]
        else:
            c_cached = c_kv

        kv = self.kv_b_proj(self.kv_a_norm(c_cached))
        s = kv.shape[1]
        kv = kv.view(b, s, self.n_heads, self.qk_nope + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(b, self.n_heads, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        out = self._sdpa(q, k, v, past, causal)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), cache


def build_attention(cfg) -> nn.Module:
    if cfg.attn_impl == "gqa":
        return GQAttention(cfg)
    if cfg.attn_impl == "mla":
        return MLAttention(cfg)
    raise ValueError(f"unknown attn_impl: {cfg.attn_impl!r}")
