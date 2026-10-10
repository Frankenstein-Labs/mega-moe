# AGENTS.md

Repository context for AI agents working on MEGA.

## What this is

A from-scratch reference implementation of a Mixture-of-Experts multimodal
decoder-only LLM (DeepSeek-V3 / Qwen3 style), plus the arithmetic needed to
decide whether a configuration can be trained. It runs on CPU.

## Commands

```bash
pip install -e ".[dev]"                 # install (CPU is fine)
python -m pytest tests/ -q              # 42 tests, ~4 min on 4 CPU cores
python -m mega.train.loop --steps 60    # smoke-train the small preset
mega-info --preset small                # parameters / memory / FLOPs / plan
mega-info --config configs/moe-1t.yaml --gpus 2048 --gpu-mem 141 --tokens 20e12
python -c "from mega.utils.feasibility import report; report()"
```

`mega-info` is installed into `~/.local/bin`; add it to `PATH` if the entry point
is not found.

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
