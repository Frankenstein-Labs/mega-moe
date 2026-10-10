import math

import pytest
import torch

from mega.model.rope import apply_rope, build_rope_cache, yarn_mscale


def test_cache_shape_and_mscale():
    cos, sin = build_rope_cache(16, 32, base=10000.0)
    assert cos.shape == (32, 16)
    assert sin.shape == (32, 16)
    # identity: cos(x)^2 + sin(x)^2 == 1
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)


def test_linear_scaling_halves_frequency():
    _, _ = build_rope_cache(8, 16, scaling={"type": "linear", "factor": 2.0})
    base = build_rope_cache(8, 16)[0]
    scaled = build_rope_cache(8, 16, scaling={"type": "linear", "factor": 2.0})[0]
    # position 1 under factor 2 should equal position 0.5 == position 0's angle start
    assert not torch.allclose(base[1], scaled[1])


def test_yarn_mscale_positive():
    assert yarn_mscale(8.0) > 1.0


def test_apply_rope_preserves_norm():
    x = torch.randn(2, 4, 10, 16)
    cos, sin = build_rope_cache(16, 10)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_apply_rope_position_zero_is_identity():
    x = torch.randn(1, 1, 5, 8)
    cos, sin = build_rope_cache(8, 5)
    # at position 0 the angle is 0 -> cos=1, sin=0
    y = apply_rope(x[:, :, :1], cos[:1], sin[:1])
    assert torch.allclose(y, x[:, :, :1], atol=1e-6)


def test_rotation_at_known_angle():
    # head_dim=2 -> single pair; angle at position p is p * base^0 = p
    cos, sin = build_rope_cache(2, 4, base=1.0)  # base=1 => freq=1 rad per position
    x = torch.tensor([[[[1.0, 0.0]]]])  # [1,1,1,2]
    y = apply_rope(x, cos[1:2], sin[1:2])
    expected = torch.tensor([math.cos(1.0), math.sin(1.0)])
    assert torch.allclose(y[0, 0, 0], expected, atol=1e-5)
