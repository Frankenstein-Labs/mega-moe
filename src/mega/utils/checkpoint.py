"""Checkpointing helpers.

A checkpoint is ``(model_state, optim_state, step, rng_state, config)`` written
with ``torch.save``.  Storing the config inside the file makes every checkpoint
self-describing, which removes a whole class of "which YAML was this?" bugs.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import torch

from ..config import ModelConfig, TrainConfig

__all__ = ["save_checkpoint", "load_checkpoint", "capture_rng", "restore_rng"]


def capture_rng() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": step,
        "rng": capture_rng(),
        "model_cfg": model_cfg.to_dict(),
        "train_cfg": train_cfg.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return str(path)


def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, dict[str, Any]]:
    """Load weights (and optimiser state) in place; return ``(step, configs)``."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload.get("rng", {}))
    configs = {
        "model_cfg": payload.get("model_cfg"),
        "train_cfg": payload.get("train_cfg"),
    }
    return int(payload.get("step", 0)), configs
