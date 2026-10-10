"""Hugging-Face-native serialisation: safetensors weights + ``config.json``.

This is the interchange format.  A training checkpoint (``utils/checkpoint.py``)
holds optimiser state and RNG so a run can resume; a ``save_pretrained``
directory holds exactly what someone needs to *use* the model: weights in
safetensors, the architecture config, and a generation config.

Keeping the two separate matters.  Checkpoints are large and short-lived;
safetensors exports are small, portable, and safe to publish (no pickle).

    from mega.hub.serialization import save_pretrained, from_pretrained
    save_pretrained(model, "out/mega-small")
    model = from_pretrained("out/mega-small")
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import ModelConfig
from ..model.transformer import MegaModel

__all__ = ["save_pretrained", "from_pretrained", "CONFIG_NAME", "WEIGHTS_NAME", "GENERATION_NAME"]

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
GENERATION_NAME = "generation_config.json"

# Keys we add to config.json for Hub readers; they are metadata, not model fields.
_HUB_META_KEYS = {"architectures", "model_type", "torch_dtype", "stage", "base_checkpoint"}


def _model_fields(raw: dict) -> dict:
    """Keep only keys :class:`ModelConfig` understands, so metadata can round-trip."""
    return {k: v for k, v in raw.items() if k not in _HUB_META_KEYS}


def save_pretrained(
    model: MegaModel,
    directory: str | Path,
    *,
    tokenizer=None,
    generation: dict | None = None,
    extra_config: dict | None = None,
) -> Path:
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    save_file(
        {k: v.contiguous().clone() for k, v in model.state_dict().items()},
        str(directory / WEIGHTS_NAME),
    )

    cfg = model.config().to_dict()
    cfg["architectures"] = ["MegaModel"]
    cfg["model_type"] = "mega"
    cfg["torch_dtype"] = "float32"
    if extra_config:
        cfg.update(extra_config)
    (directory / CONFIG_NAME).write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    gen = {"max_new_tokens": 64, "temperature": 0.8, "top_k": 50, "top_p": 0.95, "eos_token_id": 2}
    if generation:
        gen.update(generation)
    (directory / GENERATION_NAME).write_text(json.dumps(gen, indent=2), encoding="utf-8")

    if tokenizer is not None:
        tokenizer.save_pretrained(directory)

    return directory


def from_pretrained(source: str | Path, *, map_location: str | torch.device = "cpu") -> MegaModel:
    """Rebuild a model from a local ``save_pretrained`` directory or a Hub repo id."""
    from safetensors.torch import load_file

    src = Path(source)
    if not (src / CONFIG_NAME).exists():
        if src.exists():
            raise FileNotFoundError(f"{src} has no {CONFIG_NAME}")
        # a Hub repo id
        from huggingface_hub import hf_hub_download

        cfg_file = Path(hf_hub_download(repo_id=str(source), filename=CONFIG_NAME))
        weights_file = Path(hf_hub_download(repo_id=str(source), filename=WEIGHTS_NAME))
    else:
        cfg_file, weights_file = src / CONFIG_NAME, src / WEIGHTS_NAME

    cfg = ModelConfig.from_dict(_model_fields(json.loads(cfg_file.read_text(encoding="utf-8"))))
    model = MegaModel(cfg)
    state = load_file(str(weights_file), device=str(map_location))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # rope_cos / rope_sin are non-persistent and rebuilt from the config, so they
    # legitimately do not appear in the file; anything else missing is a bug.
    real_missing = [k for k in missing if not k.startswith("rope_")]
    if real_missing or unexpected:
        raise ValueError(
            f"state dict mismatch: missing={real_missing[:8]} unexpected={list(unexpected)[:8]}"
        )
    model.eval()
    return model


def load_generation_config(source: str | Path) -> dict:
    src = Path(source)
    path = src / GENERATION_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
