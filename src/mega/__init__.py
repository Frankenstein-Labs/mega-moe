"""MEGA: reference MoE multimodal language model.

The package is deliberately layered so that every component can be imported and
exercised on CPU with tiny configs:

    config      -> dataclasses describing the model / training setup
    model       -> transformer, attention (GQA + MLA), MoE, optimiser
    multimodal  -> vision / audio encoders and the fusion model
    data        -> tokenizer + packing + synthetic data
    parallel    -> parallelism planning and memory/FLOPs arithmetic
    utils       -> calculators, logging, checkpointing
"""

__version__ = "0.1.0"

from .config import ModelConfig, TrainConfig

__all__ = ["ModelConfig", "TrainConfig", "__version__"]
