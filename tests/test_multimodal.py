import torch

from mega.config import ModelConfig
from mega.multimodal.models import (
    AudioConfig,
    AudioEncoder,
    FusionConfig,
    MultimodalModel,
    PerceiverResampler,
    VisionEncoder,
    ViTConfig,
)


def text_cfg():
    return ModelConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                       ffn_hidden=128, vocab_size=256, max_seq_len=64, moe_every=2,
                       n_routed_experts=4, n_shared_experts=1, moe_top_k=2)


def test_vision_encoder_image():
    v = VisionEncoder(ViTConfig(image_size=32, patch_size=8, width=64, depth=2))
    out = v(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 16, 64)  # (32/8)^2 = 16 patches


def test_vision_encoder_video_pools_time():
    v = VisionEncoder(ViTConfig(image_size=32, patch_size=8, width=64, depth=2, temporal_pool=2))
    out = v(torch.randn(2, 6, 3, 32, 32))
    assert out.shape[0] == 2 and out.shape[2] == 64


def test_audio_encoder_downsample():
    a = AudioEncoder(AudioConfig(n_mels=80, width=64, depth=2, n_frames=100))
    out = a(torch.randn(2, 80, 100))
    assert out.shape[0] == 2 and out.shape[2] == 64


def test_resampler_fixed_token_count():
    r = PerceiverResampler(in_dim=64, out_dim=128, n_latents=8, n_heads=4, depth=2)
    assert r(torch.randn(3, 57, 64)).shape == (3, 8, 128)


def test_multimodal_forward_with_vision_and_audio():
    m = MultimodalModel(
        text_cfg(),
        FusionConfig(
            vision=ViTConfig(image_size=32, patch_size=8, width=64, depth=2, n_latents=8),
            audio=AudioConfig(n_mels=80, width=64, depth=2, n_latents=8, n_frames=100),
        ),
    )
    ids = torch.randint(1, 256, (2, 12))
    out = m(
        ids,
        labels=ids,
        pixel_values=torch.randn(2, 3, 32, 32),
        mel=torch.randn(2, 80, 100),
    )
    assert out.logits.shape == (2, 12, 256)
    assert torch.isfinite(out.loss)


def test_multimodal_text_only_matches_backbone_shape():
    m = MultimodalModel(text_cfg(), FusionConfig())
    ids = torch.randint(1, 256, (2, 10))
    out = m(ids, labels=ids)
    assert out.logits.shape == (2, 10, 256)
    assert torch.isfinite(out.loss)
