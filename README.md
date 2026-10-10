# MEGA — a Mixture-of-Experts multimodal LLM, implemented and testable

A from-scratch, runnable reference for the architecture family behind
DeepSeek-V3 / Qwen3-class models: decoder-only transformer, grouped-query and
multi-head latent attention, RoPE with YaRN, SwiGLU, and a large sparse
Mixture-of-Experts with shared experts and auxiliary-loss-free balancing, plus
vision and audio encoders feeding a shared Perceiver resampler.

It runs **on CPU in minutes** (42 tests, verified to reduce loss), and it ships
the arithmetic you need to decide whether a config can be trained at all.

> **Read [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) first if you came here for a
> 200-trillion-parameter model.** The short version: the parameter counts are
> frontier-edge but internally consistent and not absurd; the 10M-token context,
> "FP4 training with zero loss > 0.5%", and the simultaneous benchmark targets
> are the parts that are not achievable as stated, and this repo says so with
> reproducible numbers rather than pretending otherwise.

## Quick start

```bash
pip install -e ".[dev,tokenizer,hub]"

# 1. does my config fit?  (parameters, memory, FLOPs, parallel plan)
mega info --preset small
mega info --config configs/moe-1t.yaml --gpus 2048 --gpu-mem 141 --tokens 20e12

# 2. is the model correct?
python -m pytest tests/ -q

# 3. does the whole pipeline work, end to end?
bash scripts/smoke.sh          # corpus -> tokenizer -> train -> export -> eval -> generate
```

## Train a real model

A complete, reproducible run on the structured corpus (`docs/PIPELINE.md` has the
detail):

```bash
mega corpus   --out data/corpus --lines 60000
mega tokenize --input data/corpus/train.txt --model-prefix data/tokenizer/mega \
              --vocab-size 800 --model-type unigram
mega train    --config configs/mega-small.yaml --data data/corpus/train.txt \
              --tokenizer data/tokenizer/mega.model --export
mega eval     --model runs/mega-small/hf
mega generate --model runs/mega-small/hf --prompt "73 - 42 =" --max-new-tokens 8
mega push     --repo <user>/mega-small --model runs/mega-small/hf --dataset data/corpus
```

`configs/mega-small.yaml` is a real MoE (16 routed + 2 shared experts, top-2,
auxiliary-loss-free routing) trained on CPU. `mega eval` reports perplexity,
next-token accuracy and **arithmetic accuracy** — the fraction of `a op b =`
prompts the model actually gets right.

There is also SFT (`mega sft`), which reuses the same trainer with a
response-only loss mask, and a Gradio demo under `spaces/`.

## What is implemented

| component | file | notes |
|---|---|---|
| RMSNorm, SwiGLU | `model/norms.py`, `model/ffn.py` | fp32 variance, SiLU gate |
| RoPE + linear/YaRN scaling | `model/rope.py` | NTK-by-parts, `mscale` temperature |
| GQA + MLA attention | `model/attention.py` | shared KV-cache contract, torch SDPA (FlashAttention on GPU) |
| MoE | `model/moe.py` | top-k routing, shared experts, aux + z loss, learned expert bias |
| Transformer + KV-cache generation | `model/transformer.py` | cached decode verified to match full forward |
| Muon + AdamW | `model/optimizer.py` | Newton-Schulz orthogonalisation, warmup/WSD schedule |
| Vision / audio / fusion | `multimodal/models.py` | ViT + temporal pooling, Whisper-style encoder, Perceiver resampler |
| Data | `data/pack.py`, `data/text.py`, `data/sft.py` | sequence packing, structured corpus, instruction pairs with response-only mask |
| Parallelism planner | `parallel/plan.py` | 3D/4D/5D search, memory + FLOPs + KV-cache math |
| Training loop | `train/loop.py` | grad accum, clipping, checkpoint + exact resume |
| SFT | `train/sft.py` | instruction tuning over the same trainer |
| Evaluation | `eval/harness.py` | perplexity, next-token and arithmetic accuracy |
| Hub export | `hub/serialization.py`, `hub/__init__.py` | safetensors + config.json, model card, upload |
| CLI | `cli.py` | `mega {info,corpus,tokenize,train,sft,eval,generate,push}` |
| Distributed entry | `train/distributed.py` | validates topology, reports memory, no fake TP/PP |
| CI / demo | `.github/workflows/ci.yml`, `spaces/` | lint + tests + smoke; Gradio Space |

## Verified results

`python -m pytest tests/ -q` → **42 passed**, including:

* parameter-count formulas match the constructed modules **exactly**, for
  dense, MoE, tied-embedding and MLA configurations;
* incremental decoding with the KV cache reproduces the full forward pass
  (both GQA and MLA);
* causal masking holds — a future token cannot change an earlier logit;
* MoE routing weights sum to 1, expert counts are conserved, and the
  auxiliary-loss-free bias update moves busy experts down and idle ones up;
* training reduces loss (6.26 → 4.50 over 150 steps on a 4-layer CPU model);
* checkpoint save/load is numerically exact and restores the step counter.

## Reproduce the feasibility numbers

```bash
python -c "from mega.utils.feasibility import report; report()"
```

prints, for the brief's headline shape (100 layers, 40 000 experts, top-64):

```
total params        : 201.39T
active params/token : 384.25B  (ratio 524:1)
training FLOPs @20T : 65.77Y  ->  38 days on 100k H100 @ 20% MFU
optimizer state     : 3.22P bytes
KV cache  10M tokens: 1,072.9 GiB per sequence
```

## Layout

```
src/mega/
  config.py           ModelConfig / TrainConfig (JSON round-trip, validated)
  cli.py              unified `mega` command line
  model/              attention, MoE, transformer, optimizer, norms, rope, ffn
  multimodal/         vision + audio encoders, perceiver resampler, fusion
  data/               packing, structured corpus, SFT pairs
  parallel/           parameter/FLOPs/memory math, parallel-plan search
  tokenizer/          SentencePiece training + Hub-friendly wrapper
  train/              training loop, SFT, distributed entry point
  eval/               perplexity, next-token and arithmetic accuracy
  hub/                safetensors export, model card, Hub upload
  utils/              checkpointing, mega-info calculator, feasibility report
configs/              small.yaml, mega-small.yaml (trained), moe-1t.yaml
scripts/              install_cluster.sh, smoke.sh, slurm/
spaces/               Gradio demo Space
.github/workflows/    CI: lint + tests + end-to-end smoke
tests/                test suite
docs/                 PIPELINE, FEASIBILITY, ARCHITECTURE, ROADMAP, INFRASTRUCTURE
```

## Honest limitations

* **No distributed tensor/pipeline/expert-parallel compute.** `train/distributed.py`
  validates your topology and initialises the process group, then runs the
  reference loop on rank 0. Wiring Megatron-LM / DeepSpeed kernels is real
  integration work that is *not* faked here.
* **No trained large model.** The largest thing that trains here is the `small`
  preset.
* **No FP4 kernels.** `TrainConfig.quantized_linear` is a declared target, not
  an implementation.
* **Multimodal fusion prepends** soft tokens rather than interleaving them via
  placeholder ids. The cross-attention path is identical; the sequence plumbing
  is simplified.
* **MoE dispatch is an exact Python loop** over experts. Correct, and fine for
  tests; a real run needs a grouped GEMM / all-to-all kernel.

## References

DeepSeek-V3 (arXiv:2412.19437) · DeepSeek-V2 (arXiv:2405.04434) ·
DeepSeekMoE (arXiv:2401.06066) · Qwen3 (arXiv:2505.09388) · Muon
(Keller Jordan, 2024) · YaRN (arXiv:2309.00071) · GQA (arXiv:2305.13245) ·
Switch Transformers (arXiv:2101.03961).

License: Apache-2.0.
