"""Unified ``mega`` command line.

Subcommands map onto the pipeline stages, so a full run is:

    mega corpus   --out data/corpus --lines 60000
    mega tokenize --input data/corpus/train.txt --model-prefix data/tokenizer/mega --vocab-size 800
    mega train    --config configs/mega-small.yaml --data data/corpus/train.txt \
                  --tokenizer data/tokenizer/mega.model --export
    mega eval     --model runs/mega-small/hf
    mega generate --model runs/mega-small/hf --prompt "Le chat noir" --max-new-tokens 12
    mega push     --repo <user>/mega-small --model runs/mega-small/hf --dataset data/corpus

Each stage is also importable as ``python -m mega.<module>``; this CLI is the
convenience layer, not a second implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mega", description="MEGA model toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="parameter / memory / FLOPs arithmetic")

    sp = sub.add_parser("corpus", help="generate the structured text corpus")
    sp.add_argument("--out", default="data/corpus")
    sp.add_argument("--lines", type=int, default=200_000)
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("tokenize", help="train a SentencePiece tokenizer")
    sp.add_argument("--input", nargs="+", required=True)
    sp.add_argument("--model-prefix", default="data/tokenizer/mega")
    sp.add_argument("--vocab-size", type=int, default=800)
    sp.add_argument("--model-type", default="unigram", choices=["bpe", "unigram"])

    sp = sub.add_parser("train", help="pretrain on text")
    sp.add_argument("--config")
    sp.add_argument("--data")
    sp.add_argument("--tokenizer")
    sp.add_argument("--steps", type=int)
    sp.add_argument("--out")
    sp.add_argument("--export", action="store_true")

    sp = sub.add_parser("sft", help="instruction-tune a checkpoint")
    sp.add_argument("--init")
    sp.add_argument("--config")
    sp.add_argument("--tokenizer", required=True)
    sp.add_argument("--data")
    sp.add_argument("--steps", type=int, default=400)
    sp.add_argument("--out", default="runs/mega-small-sft")
    sp.add_argument("--export", default=None)

    sp = sub.add_parser("eval", help="evaluate a saved model")
    sp.add_argument("--model", required=True)
    sp.add_argument("--tokenizer", default=None)
    sp.add_argument("--lines", type=int, default=400)
    sp.add_argument("--json-out", default=None)

    sp = sub.add_parser("generate", help="sample from a saved model")
    sp.add_argument("--model", required=True)
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=32)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--top-k", type=int, default=None)

    sp = sub.add_parser("push", help="upload model/tokenizer/dataset to the Hub")
    sp.add_argument("--repo", required=True, help="e.g. user/mega-small")
    sp.add_argument("--model", required=True, help="local save_pretrained directory")
    sp.add_argument("--dataset", default=None, help="corpus directory to upload as a dataset")
    sp.add_argument("--dataset-repo", default=None)
    sp.add_argument("--token", default=None, help="defaults to $HF_TOKEN")
    sp.add_argument("--private", action="store_true")
    sp.add_argument("--eval-lines", type=int, default=0,
                    help="if >0, run the eval harness and record metrics in the model card")
    sp.add_argument("--refresh-card", action="store_true",
                    help="regenerate README.md even if the directory already has one")

    return p


def _run_info(rest: list[str]) -> int:
    from .utils.calc import main as info_main

    return info_main(rest)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return 0

    command, rest = argv[0], argv[1:]

    if command == "info":
        return _run_info(rest)
    if command == "corpus":
        from .data.text import main as m

        return m(rest)
    if command == "tokenize":
        from .tokenizer.train_tokenizer import main as m

        return m(rest)
    if command == "train":
        from .train.loop import main as m

        return m(rest)
    if command == "sft":
        from .train.sft import main as m

        return m(rest)
    if command == "eval":
        from .eval.harness import main as m

        return m(rest)
    if command == "generate":
        return _run_generate(rest)
    if command == "push":
        return _run_push(rest)

    build_parser().print_help()
    return 2


def _run_generate(argv: list[str]) -> int:
    import torch

    from .hub.serialization import from_pretrained, load_generation_config
    from .tokenizer.tokenizer import MegaTokenizer

    args = build_parser().parse_args(["generate", *argv])
    model = from_pretrained(args.model)
    tok = MegaTokenizer.from_pretrained(args.model if (Path(args.model) / "tokenizer.model").exists() else args.model)
    gen = load_generation_config(args.model)

    ids = tok.encode(args.prompt, add_bos=True)
    out = model.generate(
        torch.tensor([ids]),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or gen.get("top_k"),
        top_p=None if args.temperature <= 0 else gen.get("top_p"),
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0, len(ids):].tolist())
    print(f"prompt         : {args.prompt}")
    print(f"continuation   : {text}")
    print(f"full           : {tok.decode(out[0].tolist())}")
    return 0


def _run_push(argv: list[str]) -> int:
    import os

    from .hub import create_repo, model_card, upload_folder
    from .hub.serialization import from_pretrained

    args = build_parser().parse_args(["push", *argv])
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("no token: pass --token or set HF_TOKEN")

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"{model_dir} is not a save_pretrained directory")

    model = from_pretrained(model_dir)
    cfg = model.config()
    total = model.num_parameters()
    active = model.num_parameters(active=True)

    # attach a model card if the directory does not already have one
    readme = model_dir / "README.md"
    if not readme.exists() or args.refresh_card:
        metrics = None
        if args.eval_lines:
            from .eval.harness import evaluate

            res = evaluate(model_dir, n_lines=args.eval_lines)
            metrics = {
                "perplexity": res["perplexity"],
                "next_token_accuracy": res["next_token_accuracy"],
                "arithmetic_accuracy": res["arithmetic_accuracy"],
            }
        readme.write_text(
            model_card(
                cfg,
                total_params=total,
                active_params=active,
                tokenizer_name="SentencePiece unigram, 800 tokens",
                dataset_name=args.dataset_repo or "Frankenstein-Labs/mega-corpus",
                metrics=metrics,
                code_url="https://github.com/Frankenstein-Labs/mega-moe",
            ),
            encoding="utf-8",
        )

    create_repo(args.repo, token=token, private=args.private)
    upload_folder(args.repo, model_dir, token=token, commit_message="Upload MEGA model")
    print(f"model    : https://huggingface.co/{args.repo}")

    if args.dataset:
        ds_repo = args.dataset_repo or f"{args.repo.rsplit('/', 1)[0]}/mega-corpus"
        create_repo(ds_repo, token=token, private=args.private, exist_ok=True)
        ds_dir = Path(args.dataset)
        (ds_dir / "README.md").write_text(
            "# MEGA corpus\n\nDeterministic structured corpus generated by `mega.data.text` "
            "(bilingual templated sentences, arithmetic, word problems, code). "
            "No scraped data; Apache-2.0.\n",
            encoding="utf-8",
        )
        upload_folder(ds_repo, ds_dir, token=token, repo_type="dataset", commit_message="Upload corpus")
        print(f"dataset  : https://huggingface.co/datasets/{ds_repo}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
