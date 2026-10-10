"""Normalisation layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    Computes in float32 and casts back, which is what every large-scale
    implementation does to avoid bf16 drift in the variance estimate.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
