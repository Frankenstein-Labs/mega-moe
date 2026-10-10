"""Multimodal encoders and the fusion model.

Three encoders feed a shared Perceiver-style resampler that produces a fixed
number of soft tokens per modality.  Those tokens are concatenated with the text
embeddings and processed by one decoder-only :class:`~mega.model.transformer.MegaModel`,
so there is a single set of transformer weights and a single loss.

* **Vision / video** — a ViT encoder with learned absolute position embeddings
  and optional temporal pooling (a lightweight stand-in for TimeSformer's
  divided space-time attention).
* **Audio** — a log-mel front end feeding a small conv + transformer stack
  (Whisper-style encoder).
* **Resampler** — latent queries + cross-attention (Perceiver Resampler), which
  is what gives a fixed token budget regardless of input resolution/length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..model.norms import RMSNorm
from ..model.transformer import MegaModel

__all__ = ["ViTConfig", "VisionEncoder", "AudioConfig", "AudioEncoder",
           "PerceiverResampler", "FusionConfig", "MultimodalModel"]


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 8
    in_channels: int = 3
    width: int = 128
    depth: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    n_latents: int = 16
    temporal_pool: int = 1

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class ViTBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.width)
        self.attn = nn.MultiheadAttention(cfg.width, cfg.n_heads, batch_first=True)
        self.norm2 = RMSNorm(cfg.width)
        hidden = int(cfg.width * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.width, hidden), nn.GELU(), nn.Linear(hidden, cfg.width)
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.width, cfg.patch_size, cfg.patch_size
        )
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.width))
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.depth)])
        self.norm = RMSNorm(cfg.width)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` or ``[B, T, C, H, W]`` -> ``[B, n_patches, width]``.

        A leading time dimension is folded into the batch and then mean-pooled,
        which is the cheap temporal compression; a real TimeSformer would apply
        divided space-time attention instead.
        """
        if pixel_values.dim() == 5:
            b, t, c, h, w = pixel_values.shape
            x = self._encode(pixel_values.reshape(b * t, c, h, w))
            x = x.view(b, t, x.shape[1], x.shape[2])
            if self.cfg.temporal_pool > 1:
                x = F.avg_pool2d(
                    x.transpose(1, 3), (self.cfg.temporal_pool, 1)
                ).transpose(1, 3)
            return x.mean(dim=1)
        return self._encode(pixel_values)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


@dataclass
class AudioConfig:
    n_mels: int = 80
    width: int = 128
    depth: int = 4
    n_heads: int = 4
    conv_kernel: int = 3
    n_latents: int = 16
    n_frames: int = 100


class AudioEncoder(nn.Module):
    """Log-mel + conv downsampling + transformer (Whisper-style encoder)."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(cfg.n_mels, cfg.width, cfg.conv_kernel, padding=1)
        self.conv2 = nn.Conv1d(cfg.width, cfg.width, cfg.conv_kernel, stride=2, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_frames // 2, cfg.width))
        layer = nn.TransformerEncoderLayer(
            cfg.width, cfg.n_heads, dim_feedforward=4 * cfg.width,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.depth)
        self.norm = RMSNorm(cfg.width)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``[B, n_mels, frames]`` -> ``[B, frames/2, width]``."""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        x = x + self.pos[:, : x.shape[1]]
        return self.norm(self.enc(x))


class PerceiverResampler(nn.Module):
    """Fixed-size bottleneck from an arbitrary-length modality sequence."""

    def __init__(self, in_dim: int, out_dim: int, n_latents: int = 16, n_heads: int = 4, depth: int = 2):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(1, n_latents, out_dim))
        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.blocks = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm_q": RMSNorm(out_dim),
                    "norm_kv": RMSNorm(out_dim),
                    "cross": nn.MultiheadAttention(out_dim, n_heads, batch_first=True),
                    "norm_ff": RMSNorm(out_dim),
                    "ff": nn.Sequential(
                        nn.Linear(out_dim, 4 * out_dim), nn.GELU(),
                        nn.Linear(4 * out_dim, out_dim),
                    ),
                }
            )
            for _ in range(depth)
        ])
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        lat = self.latents.expand(x.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk["norm_q"](lat)
            kv = blk["norm_kv"](x)
            a, _ = blk["cross"](q, kv, kv, need_weights=False)
            lat = lat + a
            lat = lat + blk["ff"](blk["norm_ff"](lat))
        return lat


@dataclass
class FusionConfig:
    vision: ViTConfig = field(default_factory=ViTConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video_tokens: int = 16


class MultimodalModel(nn.Module):
    """Text backbone + vision/audio encoders + resamplers + fusion head.

    Modality embeddings are learned so that the backbone can tell a soft token
    from a text token.  Everything shares one loss on the text stream; modality
    alignment is implicit through next-token prediction over interleaved
    sequences.
    """

    def __init__(self, text_cfg: ModelConfig, fusion_cfg: FusionConfig = FusionConfig()):
        super().__init__()
        self.text = MegaModel(text_cfg)
        self.fusion_cfg = fusion_cfg
        d = text_cfg.d_model

        self.vision = VisionEncoder(fusion_cfg.vision)
        self.vision_resampler = PerceiverResampler(
            fusion_cfg.vision.width, d, fusion_cfg.vision.n_latents
        )
        self.audio = AudioEncoder(fusion_cfg.audio)
        self.audio_resampler = PerceiverResampler(
            fusion_cfg.audio.width, d, fusion_cfg.audio.n_latents
        )
        self.modality_embed = nn.Parameter(torch.zeros(3, d))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision(pixel_values)
        soft = self.vision_resampler(feats)
        return soft + self.modality_embed[0]

    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        feats = self.audio(mel)
        soft = self.audio_resampler(feats)
        return soft + self.modality_embed[1]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
    ):
        """Interleave modality soft tokens before the text embeddings.

        Soft tokens are prepended; a production model would use placeholder
        token ids and ``masked_scatter`` to interleave them anywhere in the
        sequence.  Prepending keeps the reference implementation short while
        exercising exactly the same cross-attention path.
        """
        parts = []
        if pixel_values is not None:
            parts.append(self.encode_vision(pixel_values))
        if mel is not None:
            parts.append(self.encode_audio(mel))

        text_emb = self.text.embed(input_ids)
        if not parts:
            return self.text(input_ids, labels=labels)

        prefix = torch.cat(parts, dim=1)
        x = torch.cat([prefix, text_emb], dim=1)
        x = x + 0  # keep dtype/grad path uniform

        # run the backbone manually so we can splice the prefix
        cos_full = self.text.rope_cos[: x.shape[1] + 1]
        sin_full = self.text.rope_sin[: x.shape[1] + 1]
        from ..model.attention import KVCache  # noqa: F401

        h = x
        stats = []
        for blk in self.text.blocks:
            h, _, st = blk(h, cos_full[: x.shape[1]], sin_full[: x.shape[1]], causal=True)
            if st is not None:
                stats.append(st)
        h = self.text.final_norm(h)
        logits = self.text.lm_head(h)[:, prefix.shape[1]:]  # drop soft-token logits

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        aux = torch.stack([s.aux_loss for s in stats]).mean() if stats else None
        from ..model.transformer import ModelOutput

        return ModelOutput(logits=logits, loss=loss, aux_loss=aux)
