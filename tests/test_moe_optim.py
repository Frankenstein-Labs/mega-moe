import pytest
import torch

from mega.config import ModelConfig, TrainConfig
from mega.model.moe import MoELayer
from mega.model.optimizer import MuonAdamW, build_optimizer, lr_at, split_params
from mega.model.transformer import MegaModel


def cfg(**kw):
    base = dict(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, ffn_hidden=128,
        vocab_size=256, max_seq_len=32, moe_every=1, n_routed_experts=6,
        n_shared_experts=1, moe_top_k=2, init_std=0.02,
    )
    base.update(kw)
    return ModelConfig(**base)


# ------------------------------------------------------------------ MoE
def test_moe_topk_weights_sum_to_one():
    layer = MoELayer(cfg())
    x = torch.randn(20, 64)
    idx, w, logits = layer.route(x)
    assert idx.shape == (20, 2)
    assert torch.allclose(w.sum(-1), torch.ones(20), atol=1e-5)


def test_moe_forward_shape_and_stats():
    layer = MoELayer(cfg())
    x = torch.randn(2, 8, 64)
    out, stats = layer(x)
    assert out.shape == x.shape
    assert stats.tokens_per_expert.sum() == 2 * 8 * 2  # batch*seq*top_k
    assert torch.isfinite(stats.aux_loss)


def test_expert_bias_update_moves_toward_balance():
    layer = MoELayer(cfg(router_bias=True))
    counts = torch.tensor([100, 0, 0, 0, 0, 0])
    before = layer.expert_bias.clone()
    layer.update_expert_bias(counts, gamma=0.1)
    # busy expert pushed down, idle experts pushed up
    assert layer.expert_bias[0] < 0 < layer.expert_bias[1]
    assert not torch.equal(before, layer.expert_bias)


def test_sigmoid_router():
    layer = MoELayer(cfg(router_score="sigmoid"))
    x = torch.randn(4, 64)
    idx, w, logits = layer.route(x)
    assert torch.allclose(w.sum(-1), torch.ones(4), atol=1e-5)


def test_shared_experts_add_parameters():
    a = MoELayer(cfg(n_shared_experts=0))
    b = MoELayer(cfg(n_shared_experts=2))
    na = sum(p.numel() for p in a.parameters())
    nb = sum(p.numel() for p in b.parameters())
    assert nb > na


# ------------------------------------------------------------------ optimiser
def test_split_params_separates_2d_matrices():
    model = MegaModel(cfg())
    muon, adamw = split_params(model)
    names = {n for n, _ in model.named_parameters()}
    assert muon and adamw
    assert all(p.ndim == 2 for p in muon)
    # embedding and router must never be in the Muon group
    muon_ids = {id(p) for p in muon}
    for n, p in model.named_parameters():
        if "embed" in n or "router" in n or "lm_head" in n:
            assert id(p) not in muon_ids


def test_muon_step_reduces_loss_on_quadratic():
    torch.manual_seed(0)
    target = torch.randn(16, 16)
    w = torch.nn.Parameter(torch.zeros(16, 16))

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = w

    opt = MuonAdamW(Holder(), muon_lr=0.05)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_adamw_path_updates_parameters():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(8))
    opt = torch.optim.AdamW([p], lr=0.1)
    p.grad = torch.ones(8)
    opt.step()
    assert not torch.allclose(p.detach(), torch.zeros(8))


def test_lr_schedule_warmup_then_peak_then_decay():
    tc = TrainConfig(lr_schedule="wsd", warmup_steps=10, max_steps=100, wsd_decay_frac=0.2)
    assert lr_at(0, tc) < lr_at(5, tc) < lr_at(9, tc)
    assert lr_at(10, tc) == pytest.approx(1.0)
    assert lr_at(80, tc) == pytest.approx(1.0)
    assert lr_at(99, tc) < 1.0


def test_cosine_schedule_reaches_floor():
    tc = TrainConfig(lr_schedule="cosine", warmup_steps=0, max_steps=100, min_lr_ratio=0.1)
    assert lr_at(100, tc) == pytest.approx(0.1, abs=1e-6)


def test_build_optimizer_adamw_variant():
    model = MegaModel(cfg())
    tc = TrainConfig(optimizer="adamw")
    opt = build_optimizer(model, tc)
    assert isinstance(opt, torch.optim.AdamW)
