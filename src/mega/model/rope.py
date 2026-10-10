"""Rotary position embeddings with linear and YaRN context extension.

The cache holds ``cos``/``sin`` of shape ``[max_seq_len, head_dim]`` where the
frequency for dimension pair ``i`` is duplicated across both halves, so it pairs
with the LLaMA "rotate half" convention: ``x[:d/2]`` and ``x[d/2:]`` are rotated
against each other.  This preserves the per-head norm exactly.

YaRN (Peng et al., 2023) is implemented as an NTK-by-parts frequency
interpolation: high frequencies are left untouched, low frequencies are divided
by the extension factor, and the two regimes are blended with a linear ramp.
"""

from __future__ import annotations

import math

import torch

__all__ = ["build_rope_cache", "apply_rope", "yarn_mscale"]


def _linear_ramp_factor(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    idx = torch.arange(dim, dtype=torch.float32)
    return torch.clamp((idx - low) / (high - low), 0.0, 1.0)


def _correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_mscale(factor: float) -> float:
    """YaRN attention temperature compensation."""
    return 0.1 * math.log(factor) + 1.0


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
    scaling: dict | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` each of shape ``[max_seq_len, head_dim]``."""
    half = head_dim // 2
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    inv_freq = 1.0 / (base**idx)

    mscale = 1.0
    if scaling:
        stype = scaling.get("type", "linear")
        factor = float(scaling.get("factor", 1.0))
        if stype == "linear":
            inv_freq = inv_freq / factor
        elif stype == "yarn":
            orig_max = int(scaling.get("original_max_seq_len", max_seq_len))
            beta_fast = float(scaling.get("beta_fast", 32.0))
            beta_slow = float(scaling.get("beta_slow", 1.0))
            low = max(math.floor(_correction_dim(beta_fast, head_dim, base, orig_max)), 0)
            high = min(math.ceil(_correction_dim(beta_slow, head_dim, base, orig_max)), half - 1)
            ramp = _linear_ramp_factor(low, high, half).to(device)
            inv_freq = inv_freq / factor * (1 - ramp) + inv_freq * ramp
            mscale = yarn_mscale(factor)
        elif stype in (None, "none"):
            pass
        else:
            raise ValueError(f"unknown rope scaling type: {stype!r}")

    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)  # [S, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
    return (emb.cos() * mscale).to(dtype), (emb.sin() * mscale).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension (LLaMA convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to ``x`` of shape ``[..., T, head_dim]``.

    ``cos``/``sin`` are ``[T, head_dim]`` (already sliced to the right window).
    """
    return x * cos + _rotate_half(x) * sin
