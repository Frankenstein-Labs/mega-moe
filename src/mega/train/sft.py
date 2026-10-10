"""Instruction tuning (SFT) on top of a pretrained MEGA checkpoint.

Reuses :class:`mega.train.loop.Trainer` unchanged -- the only difference between
pretraining and SFT here is the corpus (instruction/response pairs with a
response-only loss mask) and a gentler learning rate.  That is deliberate: SFT
should not require a second training implementation.

    python -m mega.data.sft --out data/sft/train.txt --n 8000
    python -m mega.train.sft --init runs/mega-small/final.pt \
        --tokenizer data/tokenizer/mega.model --data data/sft/train.txt \
        --steps 400 --export runs/mega-small-sft/hf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ModelConfig, TrainConfig, load_yaml
from ..data.sft import SFTCorpus
from ..hub.serialization import save_pretrained
from ..train.loop import Trainer
from ..utils.checkpoint import load_checkpoint

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT a MEGA checkpoint on instructions")
    p.add_argument("--init", help="pretrained checkpoint (.pt) or safetensors dir")
    p.add_argument("--config", help="YAML with model:/train: sections (optional)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", help="Q/A text file; generated if omitted")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="runs/mega-small-sft")
    p.add_argument("--export", default=None, help="directory for a safetensors export")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..tokenizer.tokenizer import MegaTokenizer

    tok = MegaTokenizer(args.tokenizer)

    if args.config:
        raw = load_yaml(args.config)
        model_cfg = ModelConfig.from_dict(raw.get("model", {}))
        train_cfg = TrainConfig.from_dict(raw.get("train", {}))
    else:
        model_cfg, train_cfg = ModelConfig(), TrainConfig()

    # SFT settings: gentler LR, short schedule, no warmup-heavy ramp.
    train_cfg.lr = args.lr
    train_cfg.muon_lr = max(min(args.lr * 10, 0.05), 0.01)
    train_cfg.warmup_steps = max(10, args.steps // 20)
    train_cfg.max_steps = args.steps
    train_cfg.micro_batch_size = args.batch_size
    train_cfg.seq_len = args.seq_len
    train_cfg.out_dir = args.out
    train_cfg.seed = args.seed
    train_cfg.dtype = "fp32"
    train_cfg.log_every = max(10, args.steps // 10)
    train_cfg.save_every = 0

    model_cfg.vocab_size = tok.vocab_size
    model_cfg.max_seq_len = max(model_cfg.max_seq_len, args.seq_len)

    trainer = Trainer(model_cfg, train_cfg)

    if args.init:
        init = Path(args.init)
        if init.is_dir():
            from ..hub.serialization import from_pretrained

            trainer.model.load_state_dict(from_pretrained(init).state_dict())
            print(f"initialised from safetensors export {init}")
        else:
            step, _ = load_checkpoint(init, trainer.model)
            print(f"initialised from checkpoint {init} (step {step})")

    if args.data:
        data_path = Path(args.data)
    else:
        from ..data.sft import write_sft

        n = max(400, args.steps * args.batch_size)
        data_path = write_sft(Path(train_cfg.out_dir) / "sft_data.txt", n, seed=args.seed)
        print(f"generated SFT data: {data_path}")

    corpus = SFTCorpus(data_path, tok, seq_len=train_cfg.seq_len)
    print(f"SFT: {len(corpus)} pairs, seq_len={train_cfg.seq_len}")
    trainer.train(corpus)

    out = Path(train_cfg.out_dir) / "sft_final.pt"
    trainer.save(str(out))
    print(f"saved SFT checkpoint to {out}")

    export = args.export or (Path(train_cfg.out_dir) / "hf")
    save_pretrained(
        trainer.model,
        export,
        tokenizer=tok,
        extra_config={"stage": "sft", "base_checkpoint": str(args.init)},
    )
    print(f"saved safetensors export to {export}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
