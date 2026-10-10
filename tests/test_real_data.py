"""Tests for the real-data pipeline: corpus, tokenizer, SFT data, HF round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mega.config import ModelConfig
from mega.data.sft import SFTCorpus, generate_sft_pairs, write_sft
from mega.data.text import TextCorpus, generate_corpus, write_corpus
from mega.model.transformer import MegaModel

sentencepiece = pytest.importorskip("sentencepiece")

from mega.tokenizer.tokenizer import EOS_ID, MegaTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory) -> MegaTokenizer:
    import sentencepiece as spm

    d = tmp_path_factory.mktemp("tok")
    txt = d / "corpus.txt"
    txt.write_text("\n".join(generate_corpus(4000, seed=1)), encoding="utf-8")
    prefix = d / "tok"
    spm.SentencePieceTrainer.train(
        input=str(txt),
        model_prefix=str(prefix),
        vocab_size=400,
        model_type="unigram",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=["<image>", "<audio>", "<video>"],
        byte_fallback=True,
    )
    return MegaTokenizer(d / "tok.model")


def test_corpus_deterministic():
    assert generate_corpus(50, seed=7) == generate_corpus(50, seed=7)
    assert generate_corpus(50, seed=7) != generate_corpus(50, seed=8)


def test_write_corpus_splits(tmp_path):
    train, val = write_corpus(tmp_path, 1000, seed=0)
    assert train.exists() and val.exists()
    assert sum(1 for _ in train.open()) > sum(1 for _ in val.open())


def test_tokenizer_roundtrip(tiny_tokenizer):
    for text in ["Le chat noir mange une pomme .", "73 - 42 = 31", "def add(a, b): return a + b"]:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text
    assert tiny_tokenizer.vocab_size <= 400


def test_tokenizer_save_load(tmp_path, tiny_tokenizer):
    tiny_tokenizer.save_pretrained(tmp_path)
    assert (tmp_path / "tokenizer.model").exists()
    assert (tmp_path / "tokenizer_config.json").exists()
    reloaded = MegaTokenizer.from_pretrained(tmp_path)
    assert reloaded.decode(reloaded.encode("Alice has 3 apples")) == "Alice has 3 apples"


def test_text_corpus_batches(tiny_tokenizer):
    corpus = TextCorpus(
        _write_temp_corpus(), tiny_tokenizer, seq_len=32, eos_id=EOS_ID
    )
    assert len(corpus) > 0
    x, y = corpus.batch(4, 1)
    assert x.shape == y.shape == (4, 32)
    assert x.dtype == torch.long


def _write_temp_corpus() -> Path:
    import tempfile

    p = Path(tempfile.mkdtemp()) / "c.txt"
    p.write_text("\n".join(generate_corpus(500, seed=3)), encoding="utf-8")
    return p


def test_sft_mask_matches_response(tiny_tokenizer):
    pairs = generate_sft_pairs(200, seed=0)
    assert all(q and a for q, a in pairs)
    path = Path(tempfile_dir()) / "sft.txt"
    write_sft(path, 200, seed=0)
    corpus = SFTCorpus(path, tiny_tokenizer, seq_len=64)
    x, labels = corpus.batch(4, 0)
    assert x.shape == labels.shape == (4, 64)
    # every row must contain at least one supervised response token
    assert (labels != -100).any(dim=1).all()
    # the mask must start somewhere after position 0 (prompt is masked)
    assert (labels[:, 0] == -100).all()


def tempfile_dir() -> str:
    import tempfile

    return tempfile.mkdtemp()


def test_hf_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    from mega.hub.serialization import from_pretrained, save_pretrained

    cfg = ModelConfig(
        vocab_size=128, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, n_kv_heads=2,
        head_dim=8, ffn_hidden=64, n_routed_experts=4, n_shared_experts=1, moe_top_k=2,
    )
    torch.manual_seed(0)
    model = MegaModel(cfg)
    save_pretrained(model, tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()
    reloaded = from_pretrained(tmp_path)
    assert reloaded.config().d_model == cfg.d_model
    a = model(torch.tensor([[1, 2, 3]])).logits
    b = reloaded(torch.tensor([[1, 2, 3]])).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_hf_roundtrip_moe_bitwise(tmp_path):
    """MoE weights must survive the export exactly -- routing depends on them."""
    pytest.importorskip("safetensors")
    from mega.hub.serialization import from_pretrained, save_pretrained

    cfg = ModelConfig(
        vocab_size=64, max_seq_len=16, n_layers=4, d_model=32, n_heads=4, n_kv_heads=2,
        head_dim=8, ffn_hidden=64, moe_every=1, n_routed_experts=8, moe_top_k=2,
        router_bias=True,
    )
    torch.manual_seed(1)
    model = MegaModel(cfg)
    save_pretrained(model, tmp_path)
    reloaded = from_pretrained(tmp_path)
    for (n1, p1), (_n2, p2) in zip(
        model.state_dict().items(), reloaded.state_dict().items(), strict=True
    ):
        if n1.startswith("rope_"):
            continue
        assert torch.equal(p1, p2), f"mismatch at {n1}"


def test_export_detects_corrupt_state(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    from mega.hub.serialization import from_pretrained, save_pretrained

    cfg = ModelConfig(
        vocab_size=64, max_seq_len=16, n_layers=1, d_model=32, n_heads=4, n_kv_heads=2,
        head_dim=8, ffn_hidden=64, use_moe=False,
    )
    model = MegaModel(cfg)
    save_pretrained(model, tmp_path)
    state = load_file(str(tmp_path / "model.safetensors"))
    state.pop("lm_head.weight")
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="state dict mismatch"):
        from_pretrained(tmp_path)


def test_config_metadata_survives_roundtrip(tmp_path):
    """Metadata keys in config.json must not be mistaken for ModelConfig fields."""
    import json

    pytest.importorskip("safetensors")
    from mega.hub.serialization import from_pretrained, save_pretrained

    cfg = ModelConfig(
        vocab_size=64, max_seq_len=16, n_layers=1, d_model=32, n_heads=4, n_kv_heads=2,
        head_dim=8, ffn_hidden=64, use_moe=False,
    )
    save_pretrained(model := MegaModel(cfg), tmp_path, extra_config={"stage": "sft"})
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["model_type"] == "mega"
    assert raw["stage"] == "sft"
    reloaded = from_pretrained(tmp_path)  # must not raise on the metadata keys
    assert reloaded.config().d_model == 32
    assert model is not None


def test_config_json_has_all_model_fields(tmp_path):
    import json

    pytest.importorskip("safetensors")
    from mega.hub.serialization import save_pretrained

    cfg = ModelConfig(vocab_size=64, max_seq_len=16, n_layers=1, d_model=32, n_heads=4,
                      n_kv_heads=2, head_dim=8, ffn_hidden=64, use_moe=False)
    save_pretrained(MegaModel(cfg), tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    for field_name in cfg.to_dict():
        assert field_name in raw, f"config.json lost {field_name}"


def test_eval_harness_runs(tmp_path, tiny_tokenizer):
    from mega.eval.harness import arithmetic_accuracy, perplexity

    cfg = ModelConfig(
        vocab_size=tiny_tokenizer.vocab_size, max_seq_len=32, n_layers=2, d_model=32,
        n_heads=4, n_kv_heads=2, head_dim=8, ffn_hidden=64, n_routed_experts=4,
        n_shared_experts=1, moe_top_k=2,
    )
    model = MegaModel(cfg)
    docs = [[tiny_tokenizer.bos_token_id, *tiny_tokenizer.encode(ln)]
            for ln in generate_corpus(40, seed=5)]
    ppl, ntp = perplexity(model, docs, seq_len=32)
    assert ppl > 0
    assert 0.0 <= ntp <= 1.0
    acc = arithmetic_accuracy(model, tiny_tokenizer, n_samples=4)
    assert 0.0 <= acc <= 1.0


def test_model_card_and_transformers_config(tmp_path):
    from mega.hub import model_card, transformers_config

    cfg = ModelConfig(
        vocab_size=64, max_seq_len=16, n_layers=2, d_model=32, n_heads=4, n_kv_heads=2,
        head_dim=8, ffn_hidden=64, n_routed_experts=4, moe_top_k=2,
    )
    tc = transformers_config(cfg)
    assert tc["num_hidden_layers"] == 2
    assert tc["num_experts"] == 4
    card = model_card(cfg, total_params=1234, active_params=567, metrics={"ppl": 3.2})
    assert "Research" in card or "research" in card
    assert "mega" in card
    assert "ppl" in card
