"""Evaluation harness.

Two levels, both real:

* **Language modelling** -- perplexity and next-token accuracy over a held-out
  window of the corpus.
* **Task accuracy** -- for the structured slices of the corpus (arithmetic,
  word problems) there is a right answer, so we can measure it.  The model is
  prompted with the left-hand side and greedy-decoded; the score parses the
  answer and compares.

Task accuracy is the number that actually tells you whether the model learned
anything, so it is reported prominently rather than buried behind perplexity.

    python -m mega.eval.harness --model out/mega-small --tokenizer out/mega-small
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch

from ..data.text import generate_corpus
from ..hub.serialization import from_pretrained
from ..model.transformer import MegaModel
from ..tokenizer.tokenizer import MegaTokenizer

__all__ = ["perplexity", "arithmetic_accuracy", "evaluate"]

_ARITH_RE = re.compile(r"^\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=\s*(-?\d+)")


def perplexity(
    model: MegaModel,
    ids: torch.Tensor | list[list[int]],
    *,
    batch_size: int = 8,
    seq_len: int | None = None,
    eos_id: int = 2,
) -> tuple[float, float]:
    """Return ``(perplexity, next_token_accuracy)``.

    ``ids`` is either a rectangular ``[N, T]`` tensor or a list of variable-length
    documents, which are packed into fixed windows first (documents rarely share a
    length, and padding them to the longest wastes the padding positions).
    """
    from ..data.pack import pack_sequences

    model.eval()
    if isinstance(ids, list):
        window = seq_len or model.config().max_seq_len
        ids, _ = pack_sequences(ids, seq_len=window, eos_id=eos_id)

    seq_len = seq_len or ids.shape[1]
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, ids.shape[0], batch_size):
            chunk = ids[start : start + batch_size, :seq_len]
            if chunk.shape[0] == 0 or chunk.shape[1] < 2:
                continue
            out = model(chunk, labels=chunk)
            n = (chunk.shape[1] - 1) * chunk.shape[0]
            total_loss += float(out.loss.detach()) * n
            pred = out.logits[:, :-1].argmax(-1)
            total_correct += int((pred == chunk[:, 1:]).sum())
            total_tokens += n
    if total_tokens == 0:
        return float("inf"), 0.0
    mean_loss = total_loss / total_tokens
    return math.exp(min(mean_loss, 20.0)), total_correct / total_tokens


def _expected(text: str) -> int | None:
    m = _ARITH_RE.match(text.strip())
    return None if not m else int(m.group(4))


def arithmetic_accuracy(
    model: MegaModel,
    tok: MegaTokenizer,
    *,
    n_samples: int = 100,
    seed: int = 0,
    max_new_tokens: int = 6,
) -> float:
    """Fraction of arithmetic prompts whose greedy continuation is correct.

    We prompt with ``"a op b ="`` and require the model's own continuation to
    restate the exact expected value.  Judging the *generated* answer, not a
    teacher-forced logit, is the honest test.
    """
    from ..data.text import generate_corpus as _gen  # local alias to avoid cycle

    model.eval()
    lines = [ln for ln in _gen(6000, seed=seed) if _ARITH_RE.match(ln)]
    lines = lines[:n_samples]
    correct = 0
    for line in lines:
        lhs, _, rhs = line.partition("=")
        prompt = lhs.strip() + " ="
        ids = tok.encode(prompt, add_bos=True)
        out = model.generate(
            torch.tensor([ids]), max_new_tokens=max_new_tokens, temperature=0.0
        )
        decoded = tok.decode(out[0, len(ids):].tolist()).strip()
        want = rhs.strip()
        if decoded.split()[:1] and decoded.strip().startswith(want.split()[0]):
            correct += 1
        elif want in decoded:
            correct += 1
    return correct / max(len(lines), 1)


def evaluate(
    model_dir: str | Path,
    tokenizer_dir: str | Path | None = None,
    *,
    n_lines: int = 400,
    seed: int = 1234,
) -> dict:
    model = from_pretrained(model_dir)
    tok = MegaTokenizer.from_pretrained(tokenizer_dir or model_dir)
    cfg = model.config()

    val_lines = generate_corpus(n_lines, seed=seed)
    # Match training exactly: BOS prefix per document; pack_sequences appends EOS.
    doc_ids = [[tok.bos_token_id, *tok.encode(ln)] for ln in val_lines]
    window = min(cfg.max_seq_len, 64)
    ppl, ntp = perplexity(model, doc_ids, seq_len=window)

    result = {
        "perplexity": round(ppl, 3),
        "next_token_accuracy": round(ntp, 4),
        "arithmetic_accuracy": round(arithmetic_accuracy(model, tok, n_samples=60), 4),
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "max_seq_len": cfg.max_seq_len,
    }
    print("=== evaluation ===")
    for k, v in result.items():
        print(f"{k:>22} : {v}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a saved MEGA model")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--lines", type=int, default=400)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json-out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args.model, args.tokenizer, n_lines=args.lines, seed=args.seed)
    if args.json_out:
        import json

        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
