"""Training loop for the MEGA model.

Deliberately single-process and dependency-free so it runs anywhere (including
the CPU box this repo was built on).  It implements the pieces that matter and
that are easy to get wrong:

* gradient accumulation with correct loss scaling;
* global gradient-norm clipping;
* Muon+AdamW with a warmup + WSD/cosine schedule;
* MoE auxiliary/z losses added to the LM loss;
* auxiliary-loss-free expert-bias updates;
* periodic checkpointing and exact resume (optimiser state + RNG).

Distributed training is launched through the SLURM/Megatron configs in
``configs/``; this module is the reference the distributed path must match.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch

from ..config import ModelConfig, TrainConfig, configs_from_yaml
from ..data.pack import SyntheticCorpus
from ..model.optimizer import build_optimizer, lr_at
from ..model.transformer import MegaModel
from ..utils.checkpoint import load_checkpoint, save_checkpoint

__all__ = ["Trainer", "main", "build_parser"]


def _autocast(cfg: TrainConfig):
    if cfg.dtype == "bf16":
        return torch.autocast("cpu" if not torch.cuda.is_available() else "cuda", dtype=torch.bfloat16)
    if cfg.dtype == "fp16":
        return torch.autocast("cpu" if not torch.cuda.is_available() else "cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MEGA on synthetic or real text")
    p.add_argument("--config", help="YAML with model:/train: sections")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data", default=None, help="text file; enables real-data training")
    p.add_argument("--tokenizer", default=None, help="SentencePiece .model path")
    p.add_argument("--export", action="store_true", help="also write a safetensors export")
    return p


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, device: str = "cpu"):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = torch.device(device)
        torch.manual_seed(train_cfg.seed)

        self.model = MegaModel(model_cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, train_cfg)
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.step = 0

    def _set_lr(self, scale: float) -> None:
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * scale

    def train(self, corpus: SyntheticCorpus) -> list[dict]:
        cfg = self.train_cfg
        self.model.train()
        history: list[dict] = []
        t0 = time.time()

        for step in range(self.step, cfg.max_steps):
            scale = lr_at(step, cfg)
            self._set_lr(scale)
            self.optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0
            for micro in range(cfg.grad_accum_steps):
                x, y = corpus.batch(cfg.micro_batch_size, index=step * cfg.grad_accum_steps + micro)
                x, y = x.to(self.device), y.to(self.device)
                with _autocast(cfg):
                    out = self.model(x, labels=y)
                    loss = out.loss
                    if out.aux_loss is not None:
                        loss = loss + out.aux_loss + out.z_loss
                (loss / cfg.grad_accum_steps).backward()
                total_loss += float(out.loss.detach()) / cfg.grad_accum_steps

            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            # auxiliary-loss-free expert-bias update, once per optimiser step
            for block in self.model.blocks:
                if getattr(block, "is_moe", False):
                    last = getattr(block.ffn, "_last_tokens_per_expert", None)
                    if last is not None:
                        block.ffn.update_expert_bias(last)

            self.optimizer.step()
            self.step = step + 1

            if step % cfg.log_every == 0:
                rec = {
                    "step": step,
                    "loss": total_loss,
                    "lr_scale": scale,
                    "elapsed_s": time.time() - t0,
                }
                history.append(rec)
                print(f"step {step:>6}  loss {total_loss:.4f}  lr_scale {scale:.3f}  "
                      f"({rec['elapsed_s']:.1f}s)")

            if cfg.save_every and (step + 1) % cfg.save_every == 0:
                path = Path(cfg.out_dir) / f"ckpt_{step + 1}.pt"
                save_checkpoint(path, self.model, self.optimizer, self.step, self.model_cfg, cfg)
                print(f"  saved {path}")

        return history

    def save(self, path: str) -> str:
        return save_checkpoint(path, self.model, self.optimizer, self.step, self.model_cfg, self.train_cfg)

    def resume(self, path: str) -> int:
        self.step, _ = load_checkpoint(path, self.model, self.optimizer)
        return self.step


def default_configs() -> tuple[ModelConfig, TrainConfig]:
    from ..utils.calc import PRESETS

    return PRESETS["small"], TrainConfig()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config:
        model_cfg, train_cfg = configs_from_yaml(args.config)
    else:
        model_cfg, train_cfg = default_configs()

    if args.steps:
        train_cfg.max_steps = args.steps
    if args.seq_len:
        train_cfg.seq_len = args.seq_len
        model_cfg.max_seq_len = max(model_cfg.max_seq_len, args.seq_len)
    if args.batch_size:
        train_cfg.micro_batch_size = args.batch_size
    if args.out:
        train_cfg.out_dir = args.out
    if args.seed is not None:
        train_cfg.seed = args.seed

    if args.data:
        from ..data.text import TextCorpus
        from ..tokenizer.tokenizer import MegaTokenizer

        if not args.tokenizer:
            raise SystemExit("--data requires --tokenizer")
        tok = MegaTokenizer(args.tokenizer)
        model_cfg.vocab_size = tok.vocab_size
        corpus = TextCorpus(args.data, tok, seq_len=train_cfg.seq_len)
        print(f"real-data training: {len(corpus):,} windows of {train_cfg.seq_len} tokens")
    else:
        corpus = SyntheticCorpus(
            model_cfg.vocab_size, train_cfg.seq_len, n_samples=512, seed=train_cfg.seed
        )

    trainer = Trainer(model_cfg, train_cfg)
    if args.resume:
        step = trainer.resume(args.resume)
        print(f"resumed from {args.resume} at step {step}")

    trainer.train(corpus)
    out = Path(train_cfg.out_dir) / "final.pt"
    trainer.save(str(out))
    print(f"saved final checkpoint to {out}")
    if args.export:
        from ..hub.serialization import save_pretrained

        export_dir = Path(train_cfg.out_dir) / "hf"
        save_pretrained(trainer.model, export_dir, tokenizer=tok if args.data else None)
        print(f"saved safetensors export to {export_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
