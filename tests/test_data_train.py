import torch

from mega.config import ModelConfig, TrainConfig
from mega.data.pack import SyntheticCorpus, pack_sequences
from mega.parallel.plan import (
    kv_cache_bytes,
    plan_parallelism,
    training_flops,
    training_memory,
)
from mega.train.loop import Trainer
from mega.utils.checkpoint import load_checkpoint, save_checkpoint


# --------------------------------------------------------------- data
def test_pack_sequences_windows():
    docs = [list(range(10)), list(range(10)), list(range(10))]
    ids, labels = pack_sequences(docs, seq_len=16, eos_id=2)
    assert ids.shape[1] == 16
    assert ids.shape[0] == (3 * 11) // 16
    assert labels.shape == ids.shape


def test_synthetic_corpus_is_learnable_pattern():
    c = SyntheticCorpus(vocab_size=100, seq_len=32, n_samples=8, seed=0, pattern_len=4)
    x, y = c.batch(2, 0)
    # pattern repeats, so token i equals token i+4 for most positions
    assert torch.equal(x[0, :28], x[0, 4:32])


# --------------------------------------------------------------- planner
def test_plan_factorises_gpu_count():
    cfg = ModelConfig(n_layers=12, d_model=512, n_heads=8, n_kv_heads=2, head_dim=64, ffn_hidden=1024)
    for n in [8, 16, 64, 256]:
        p = plan_parallelism(cfg, n, gpu_mem_gib=80)
        assert p.tp * p.pp * p.cp * p.dp == n
        assert p.tp <= 8


def test_plan_respects_nvlink_domain():
    cfg = ModelConfig(n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8, head_dim=128, ffn_hidden=28672)
    p = plan_parallelism(cfg, 512, gpu_mem_gib=80, nvlink_domain=8)
    assert p.tp <= 8


def test_kv_cache_gqa_smaller_than_mha():
    mha = ModelConfig(n_layers=10, n_heads=32, n_kv_heads=32, head_dim=128)
    gqa = ModelConfig(n_layers=10, n_heads=32, n_kv_heads=4, head_dim=128)
    assert kv_cache_bytes(gqa, 1, 1000) < kv_cache_bytes(mha, 1, 1000)


def test_kv_cache_mla_much_smaller():
    cfg = ModelConfig(n_layers=60, n_heads=64, n_kv_heads=8, head_dim=128)
    mla = ModelConfig.from_dict(cfg.to_dict())
    mla.attn_impl = "mla"
    assert kv_cache_bytes(mla, 1, 1000) < kv_cache_bytes(cfg, 1, 1000)


def test_flops_attention_share_grows_with_context():
    cfg = ModelConfig(n_layers=12, d_model=512, n_heads=8, n_kv_heads=8, head_dim=64,
                      ffn_hidden=1024, max_seq_len=128)
    short = training_flops(cfg, 10_000)
    cfg2 = ModelConfig.from_dict(cfg.to_dict())
    cfg2.max_seq_len = 8192
    long = training_flops(cfg2, 10_000)
    assert long["attention_forward"] / long["forward"] > short["attention_forward"] / short["forward"]


def test_memory_shrinks_with_more_gpus_when_sharded():
    cfg = ModelConfig(n_layers=40, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128, ffn_hidden=11008)
    a = training_memory(cfg, n_gpus=8, shard_optimizer=True)
    b = training_memory(cfg, n_gpus=64, shard_optimizer=True)
    assert b["static_per_gpu_gib"] < a["static_per_gpu_gib"]


# --------------------------------------------------------------- training + ckpt
def test_training_reduces_loss():
    mc = ModelConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                     ffn_hidden=128, vocab_size=256, max_seq_len=64, moe_every=2,
                     n_routed_experts=4, n_shared_experts=1, moe_top_k=2)
    tc = TrainConfig(lr=5e-3, muon_lr=0.05, warmup_steps=5, max_steps=40,
                     seq_len=64, micro_batch_size=8, log_every=10, save_every=0)
    corpus = SyntheticCorpus(256, 64, n_samples=64, pattern_len=8)
    tr = Trainer(mc, tc)
    h = tr.train(corpus)
    assert h[-1]["loss"] < h[0]["loss"]


def test_checkpoint_roundtrip_is_exact(tmp_path):
    mc = ModelConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                     ffn_hidden=128, vocab_size=256, max_seq_len=32, moe_every=1,
                     n_routed_experts=4, n_shared_experts=1, moe_top_k=2)
    tc = TrainConfig()
    tr = Trainer(mc, tc)
    x = torch.randint(1, 256, (2, 8))
    before = tr.model(x).logits.detach().clone()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, tr.model, tr.optimizer, step=7, model_cfg=mc, train_cfg=tc)

    tr2 = Trainer(mc, tc)
    step, cfgs = load_checkpoint(path, tr2.model, tr2.optimizer)
    after = tr2.model(x).logits.detach().clone()
    assert step == 7
    assert torch.allclose(before, after, atol=1e-6)
    assert cfgs["model_cfg"]["n_layers"] == 2


def test_grad_clipping_is_applied():
    mc = ModelConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                     ffn_hidden=128, vocab_size=256, max_seq_len=32, moe_every=2,
                     n_routed_experts=4, n_shared_experts=1, moe_top_k=2)
    tc = TrainConfig(grad_clip=1.0, max_steps=2, seq_len=32, micro_batch_size=4,
                     log_every=1, save_every=0, lr=1e-3)
    tr = Trainer(mc, tc)
    corpus = SyntheticCorpus(256, 32, n_samples=16)
    tr.train(corpus)
    total_norm = torch.nn.utils.clip_grad_norm_(tr.model.parameters(), 1e9)
    assert torch.isfinite(total_norm)
