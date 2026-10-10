import pytest
import torch

from mega.config import ModelConfig, TrainConfig
from mega.model.transformer import MegaModel
from mega.parallel.plan import active_params, total_params


def tiny(**kw):
    base = dict(
        n_layers=4, d_model=128, n_heads=8, n_kv_heads=2, head_dim=16,
        ffn_hidden=256, vocab_size=512, max_seq_len=64, moe_every=2,
        n_routed_experts=8, n_shared_experts=1, moe_top_k=2, init_std=0.02,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_total_params_formula_matches_module():
    for cfg in [tiny(), tiny(use_moe=False), tiny(tie_embeddings=True), tiny(moe_every=1)]:
        m = MegaModel(cfg)
        assert m.num_parameters() == total_params(cfg)


def test_active_params_formula_matches_module():
    for cfg in [tiny(), tiny(use_moe=False), tiny(moe_top_k=1), tiny(moe_every=1)]:
        m = MegaModel(cfg)
        assert m.num_parameters(active=True) == active_params(cfg)


def test_active_less_than_total_with_moe():
    cfg = tiny(n_routed_experts=32, moe_top_k=2)
    assert active_params(cfg) < total_params(cfg)


def test_forward_shapes_and_loss():
    cfg = tiny()
    m = MegaModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x, labels=x)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert torch.isfinite(out.aux_loss)


def test_causal_masking():
    """Changing a future token must not change an earlier position's logits."""
    cfg = tiny(use_moe=False, qk_norm=True)
    m = MegaModel(cfg).eval()
    x = torch.randint(1, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base = m(x).logits
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
        changed = m(x2).logits
    assert torch.allclose(base[0, :-1], changed[0, :-1], atol=1e-5)
    assert not torch.allclose(base[0, -1], changed[0, -1])


def test_cache_matches_full_forward():
    """Incremental decoding with a KV cache must reproduce the full forward."""
    for impl in ["gqa", "mla"]:
        cfg = tiny(attn_impl=impl, use_moe=False)
        if impl == "mla":
            cfg.q_lora_rank = 64
        m = MegaModel(cfg).eval()
        x = torch.randint(1, cfg.vocab_size, (1, 10))
        with torch.no_grad():
            full = m(x).logits
            # token-by-token with cache (must start from empty caches)
            caches = m.empty_caches()
            step_logits = []
            for i in range(x.shape[1]):
                out = m(x[:, i : i + 1], caches=caches, past_len=i)
                caches = out.caches
                step_logits.append(out.logits[:, 0])
            step = torch.stack(step_logits, dim=1)
        assert torch.allclose(full, step, atol=1e-3), f"{impl} cache mismatch"


def test_generate_shapes_and_determinism():
    cfg = tiny(use_moe=False)
    m = MegaModel(cfg).eval()
    x = torch.randint(1, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        a = m.generate(x, max_new_tokens=6, temperature=0.0)
        b = m.generate(x, max_new_tokens=6, temperature=0.0)
    assert a.shape == (2, 11)
    assert torch.equal(a, b)


def test_validate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8, n_kv_heads=2).validate()
    with pytest.raises(ValueError):
        ModelConfig(n_heads=8, n_kv_heads=3).validate()
    with pytest.raises(ValueError):
        ModelConfig(n_routed_experts=4, moe_top_k=8).validate()
