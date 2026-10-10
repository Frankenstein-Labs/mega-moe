# Roadmap: from this reference to something real

The original brief asked for a 200 T model in 42 months. This is the version I
would actually sign up for: a sequence of milestones where each one is validated
before the next is funded, and where the first three are achievable with the
code in this repository plus a modest cluster.

## M0 — Reference validated (done)

* Architecture, MoE, optimizer, training loop implemented and tested.
* 55 tests pass on CPU; loss reduces; KV-cache decode matches full forward.
* `mega-info` reports parameters, memory, FLOPs, KV cache and a parallel plan.

**Exit criterion:** `pytest` green, `mega-info` reproduces the numbers in
`docs/FEASIBILITY.md`.

## M1 — Single-node GPU training (weeks)

Wire FlashAttention-3 and fp8/bf16 autocast, train the `small` preset on one
8×H100 node on real tokenised data.

* Add a real dataloader (WebDataset shards) behind `data/`.
* Log to W&B; add a loss-curve regression test at fixed seed.
* **Exit:** loss matches the CPU reference within noise at the same seed; a
  single node sustains > 30 % MFU on a 7 B dense model.

## M2 — Dense scale-out (1-2 months)

Train a **7 B dense** model on ~1 T tokens on 64-256 GPUs using Megatron-LM or
DeepSpeed for TP/PP/ZeRO. This is where the distributed integration lands.

* Replace the Python MoE loop with a grouped GEMM (not needed yet if dense).
* **Exit:** a 7 B model that is measurably good (beats a 1 B baseline on a held
  out eval), reproducible from a single command.

## M3 — MoE at moderate scale (3-6 months)

A **~300 B total / ~20 B active** MoE, 128 K context, on a few hundred GPUs.

* MoE all-to-all + expert parallelism.
* Auxiliary-loss-free balancing validated against the aux-loss baseline.
* Ring/context parallel for the 128 K context stage.
* **Exit:** active/total parameter ratio confirmed by measurement; expert load
  balanced within 2×; the model is competitive with a dense model of equal
  *active* size at lower compute.

## M4 — Frontier-adjacent (1-2 years, serious capital)

A **1-7 T total / 100-300 B active** MoE, 128 K context, trained on 10-20 T
tokens. This is DeepSeek-V3-class and it is *the top of what is known to work*.

Pre-flight, all from this repo:

```bash
mega-info --config configs/moe-1t.yaml --gpus 2048 --gpu-mem 141 --tokens 20e12
python -c "from mega.utils.feasibility import report; report()"
```

## M5 and beyond — what to drop

Before attempting anything past M4, resolve these, because they are the parts of
the brief that do not currently work:

* **Context.** 128 K is production-viable (14 GiB KV/seq with MLA). 1 M is a
  research mode. 10 M needs 1 TiB of KV per sequence and is not a serving
  configuration. Extend context only as far as the serving stack can afford.
* **FP4 training.** Prove it on a 7 B model first, with a measured loss delta
  against bf16. Do not plan a frontier run around an unvalidated precision.
* **Quality targets.** Set them after you have a baseline, not before. "MMLU
  > 95 % and MMMU > 85 % and HumanEval > 95 % simultaneously" is a research
  ambition, not a plan.

## Post-training and serving (parallel track)

These do not need the largest model to be useful:

* SFT on multimodal instructions; LoRA/QLoRA adapters; DPO then GRPO.
* Quantization to fp8/int4 for serving; speculative decoding; KV-cache
  compression (MLA already does much of this).
* A RAG + episodic-memory layer and a perceive → reason → act loop.

## How to decide, every time

Before any run, answer these with `mega-info`:

1. Does `tp * pp * cp * dp` equal the real world size? (`distributed.py` checks.)
2. Do the optimizer states fit after sharding? (compare `static` to
   `n_gpus * gpu_mem`)
3. Is the KV cache affordable at the context you actually want to *serve*?
4. What is the FLOPs-derived wall time at a *realistic* MFU (20 % for MoE)?

If any answer is no, change the config, not the expectation.
