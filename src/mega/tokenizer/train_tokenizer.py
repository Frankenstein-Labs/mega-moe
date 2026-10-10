"""Train a SentencePiece BPE tokenizer for the text stream.

The vocabulary size is a *budget* decision, not a quality dial: a larger vocab
shrinks sequences (cheaper attention) but grows the embedding table
``vocab_size * d_model``, which for a 1M-token vocab and d_model=8192 is ~8B
parameters -- 100x the embedding of a 128k vocab.  Run ``mega-info --vocab`` to
see the trade-off for a given model before committing.

The script is deliberately streaming: it never loads the corpus into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True, help="training text files or globs")
    p.add_argument("--model-prefix", default="tokenizer/mega")
    p.add_argument("--vocab-size", type=int, default=128_000)
    p.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    p.add_argument("--character-coverage", type=float, default=0.9995)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--max-sentence-length", type=int, default=1_000_000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "sentencepiece is not installed. Install it with:\n"
            "    pip install 'mega[tokenizer]'"
        ) from exc

    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)
    try:
        spm.SentencePieceTrainer.train(
            input=",".join(args.input),
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            num_threads=args.num_threads,
            max_sentence_length=args.max_sentence_length,
            # fixed special-token ids shared with mega.tokenizer.tokenizer
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
            user_defined_symbols=["<image>", "<audio>", "<video>"],
            byte_fallback=True,
            add_dummy_prefix=True,
            remove_extra_whitespaces=True,
        )
    except RuntimeError as exc:
        # SentencePiece refuses a vocab larger than the data can fill. Surface
        # that so the caller can pick a valid size instead of guessing.
        raise SystemExit(
            f"tokenizer training failed: {exc}\n"
            "The corpus cannot fill the requested vocabulary. Lower --vocab-size "
            "or supply more varied text."
        ) from exc
    print(f"wrote {args.model_prefix}.model / .vocab  (vocab_size={args.vocab_size})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
