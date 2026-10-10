# End-to-end pipeline

This is the reproducible path from nothing to a published model. Every step runs
on CPU; the whole chain is exercised by `scripts/smoke.sh` in CI.

```bash
pip install -e ".[dev,tokenizer,hub]"
```

## 1. Generate the corpus

```bash
mega corpus --out data/corpus --lines 60000
# or: python -m mega.data.text --out data/corpus --lines 60000
```

`mega.data.text` generates a deterministic, structure-rich corpus from grammar
templates: bilingual (fr/en) sentences, integer arithmetic, word problems and
short Python snippets. It is not web data — it cannot be — but it has real
structure and a right answer for a subset of it, which is what makes the
evaluation meaningful rather than cosmetic.

## 2. Train the tokenizer

```bash
mega tokenize --input data/corpus/train.txt \
              --model-prefix data/tokenizer/mega \
              --vocab-size 800 --model-type unigram
```

Special-token ids are fixed (`<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3) and shared
between the training script and `mega.tokenizer.tokenizer`. Do not let them drift.

SentencePiece refuses a vocabulary larger than the data can fill. If it does, the
error tells you the ceiling; lower `--vocab-size` or supply more varied text.

## 3. Pretrain

```bash
mega train --config configs/mega-small.yaml \
           --data data/corpus/train.txt \
           --tokenizer data/tokenizer/mega.model \
           --export
```

This writes:

* `runs/mega-small/final.pt` — a resumable checkpoint (weights + optimiser + RNG),
* `runs/mega-small/hf/` — a `save_pretrained` directory (safetensors + `config.json`).

The two are deliberately different artefacts. See `mega/hub/serialization.py`.

## 4. Evaluate

```bash
mega eval --model runs/mega-small/hf --lines 400
```

Reports perplexity, next-token accuracy, and **arithmetic accuracy** — the
fraction of `a op b =` prompts whose greedy continuation is actually correct.
The last number is the one that tells you the model learned something.

## 5. Instruction-tune (SFT)

```bash
mega sft --init runs/mega-small/final.pt \
         --config configs/mega-small.yaml \
         --tokenizer data/tokenizer/mega.model \
         --steps 400 \
         --export runs/mega-small-sft/hf
```

SFT reuses `Trainer` unchanged; only the corpus differs. Pairs are formatted
`Q: <instruction> A: <response>` and the loss is masked over the prompt, so the
model is trained only on the response. No new special tokens are required.

## 6. Generate

```bash
mega generate --model runs/mega-small/hf \
              --prompt "Q: Calcule : 39 + 62 A:" --max-new-tokens 8
```

## 7. Publish to the Hugging Face Hub

```bash
export HF_TOKEN=...            # or pass --token
mega push --repo <user>/mega-small \
          --model runs/mega-small/hf \
          --dataset data/corpus
```

`push` creates the repo, attaches a model card if the directory has none, and
uploads the `save_pretrained` directory. With `--dataset` it also publishes the
corpus to a `-corpus` dataset repo.

## What is exported

A `save_pretrained` directory contains:

| file | purpose |
|---|---|
| `model.safetensors` | weights, no pickle |
| `config.json` | architecture + Hub metadata (`model_type`, `architectures`) |
| `generation_config.json` | default sampling parameters |
| `tokenizer.model` | SentencePiece model |
| `tokenizer_config.json` | special tokens and modality placeholders |
| `README.md` | model card |

`config.json` carries a few metadata keys that are not `ModelConfig` fields
(`model_type`, `architectures`, `torch_dtype`, and `stage` for SFT exports);
`from_pretrained` filters them out on load. There is a round-trip test for this,
including a bitwise MoE check, because routing depends on exact weights.

## The Space

`spaces/` holds a Gradio demo. It reads `MEGA_MODEL_ID` (default
`Frankenstein-Labs/mega-small`) and prints an explicit banner that the model is a
small research artifact, not an assistant. Deploy it as a Gradio Space pointed at
the published weights.
