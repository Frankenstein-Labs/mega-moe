# AGENTS.md

Repository context for AI agents working on MEGA.

## What this is

A from-scratch reference implementation of a Mixture-of-Experts multimodal
decoder-only LLM (DeepSeek-V3 / Qwen3 style), plus the arithmetic needed to
decide whether a configuration can be trained. It runs on CPU.

## Commands

```bash
pip install -e ".[dev]"                 # install (CPU is fine)
pip install -e ".[tokenizer,hub]"       # tokenizer + safetensors/Hub support
python -m pytest tests/ -q              # test suite, ~2 min on 4 CPU cores
bash scripts/smoke.sh                   # full corpus->tokenizer->train->export->eval run

# unified CLI
mega info --preset small                # parameters / memory / FLOPs / plan
mega corpus --out data/corpus --lines 60000
mega tokenize --input data/corpus/train.txt --model-prefix data/tokenizer/mega --vocab-size 800
mega train --config configs/mega-small.yaml --data data/corpus/train.txt \
           --tokenizer data/tokenizer/mega.model --export
mega eval --model runs/mega-small/hf
mega generate --model runs/mega-small/hf --prompt "73 - 42 =" --max-new-tokens 8
mega push --repo <user>/mega-small --model runs/mega-small/hf --dataset data/corpus

mega-info --config configs/moe-1t.yaml --gpus 2048 --gpu-mem 141 --tokens 20e12
python -c "from mega.utils.feasibility import report; report()"
```

`mega` and `mega-info` are installed into `~/.local/bin`; add it to `PATH` if the
entry points are not found.

## Conventions

* Python 3.10+, `from __future__ import annotations`, dataclasses for config.
* Line length 100, ruff with `E,F,I,UP,B`.
* Every formula in `parallel/plan.py` must be checkable by hand and must match
  the constructed module — there are exact-equality tests for this. If you
  change a layer, update both `ModelConfig`-derived formulas and the modules.
* Attention implementations share the signature `(x, cos, sin, cache) -> (out, cache)`.
* Tests must use real code paths, not mocks.

## Invariants that have already been broken once — do not break them again

1. **RoPE uses the half-split convention.** `_rotate_half` splits at `d/2`, not
   even/odd indices. There is a norm-preservation test.
2. **Cached decoding must match the full forward pass.** Start from
   `model.empty_caches()`; passing `cache=None` on any step breaks the chain
   silently. `test_cache_matches_full_forward` covers GQA and MLA.
3. **MoE accumulation dtype.** `index_add_` requires matching dtypes; expert
   GEMMs may run in bf16 under autocast while the accumulator is fp32. Cast the
   contribution, not the accumulator.
4. **Parameter formulas must include QK-norm and the final norm once.** These
   are exact-equality tested.
5. **`config.json` carries Hub metadata that is not `ModelConfig` state.**
   `model_type`, `architectures`, `torch_dtype`, `stage` and `base_checkpoint`
   are written by `hub/serialization.py` and must be filtered out on load
   (`_model_fields`), or `from_pretrained` raises on its own output.
6. **`rope_cos`/`rope_sin` are non-persistent.** They are rebuilt from the
   config on load and legitimately absent from `model.safetensors`; the loader
   tolerates exactly those keys and nothing else.
7. **Special-token ids are fixed** (`<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3) and
   shared by `train_tokenizer.py` and `tokenizer/tokenizer.py`.

## When asked for something bigger (100T+ parameters, 10M context)

Do not fabricate training runs, benchmarks, or checkpoints. Point at
`docs/FEASIBILITY.md`, which has reproducible arithmetic, and at
`docs/ROADMAP.md`, which proposes a validated scale-up path. The honest answer
is: parameters scale, context and unvalidated precisions do not.

## Layout

```
src/mega/model/        attention, moe, transformer, optimizer, norms, rope, ffn
src/mega/multimodal/   encoders + fusion
src/mega/parallel/     parameter/FLOPs/memory math, plan search
src/mega/train/        loop.py (runs anywhere), distributed.py (topology checks)
src/mega/utils/        checkpoint, calc (mega-info), feasibility
docs/                  FEASIBILITY, ARCHITECTURE, ROADMAP, INFRASTRUCTURE
```
