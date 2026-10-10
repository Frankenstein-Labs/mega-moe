# Feasibility: what a 200T-parameter multimodal MoE actually costs

This document is the reality check for the original brief. Every number below is
produced by the code in this repository, so you can reproduce it:

```bash
mega-info --preset moe-6t --gpus 4096 --gpu-mem 141 --tokens 15e12
python -c "from mega.utils.feasibility import report; report()"
```

The short version: **the parameter counts in the brief are near the edge of what
is buildable in 2025-2026, but three specific requirements in it are not
achievable at any budget — the 10M-token context, FP4 *training* with "zero
loss > 0.5%", and the simultaneous quality targets.** The 42-month schedule is
fantasy for the reasons in the last section.

---

## 1. The model arithmetic

A configuration that matches the brief's headline shape (100 decoder layers,
`d_model=8192`, 40 000 routed experts, top-64, two shared experts, `expert_hidden=2048`):

| quantity | value |
|---|---|
| total parameters | **201 T** |
| active parameters / token | **384 B** |
| sparse/dense ratio | 524:1 |
| experts | 40 000 routed, top-64 + 2 shared |

So the brief's "200T total / 200-400B active / 10 000+ experts / top-64" is
**internally consistent**. I want to be fair about that: it is not a nonsense
request.

## 2. Compute is the *cheap* part

Training FLOPs follow the standard `6 · P_active · N` rule (plus the attention
term, which does not reduce with sparsity):

| tokens | training FLOPs |
|---|---|
| 20 T | 6.6 × 10^25 |

On a 100 000-GPU H100 fleet at a *generous* 40 % MFU (989 TFLOP/s bf16 peak):

```
6.6e25 / (989e12 · 0.4 · 1e5) = 1.7e6 s ≈ 19 days
```

MoE all-to-all with 40 000 experts will not hold 40 % MFU; 15-25 % is realistic,
which puts a 20 T-token run at **roughly 1-2 months of wall time** on that
fleet. That is months-scale, in line with frontier training runs. Compute alone
does **not** make this impossible.

## 3. Memory is the real wall — but it is a sharding problem

Static state per parameter, at bf16 weights + fp32 master + 2 fp32 Adam moments:

```
weights (fp8)          201 TB
weights (bf16)         402 TB
gradients (bf16)       402 TB
fp32 master + moments  2.4 PB
---------------------------------
optimizer state total  ≈ 3.2 PB
```

3.2 PB does not fit in 100 000 × 141 GB = 14 PB only if you shard it: ZeRO-3
across the whole fleet is 32 GB/GPU, which fits. **The expense is the
all-gather traffic, not the capacity.** Every step moves the equivalent of the
whole model through the fabric several times.

The storage in the brief (50 PB NVMe + 500 PB HDD) is actually the *right order*
for the optimizer states of a run this size. This part of the brief is sane.

## 4. What is genuinely impossible

### 4.1 Context length (the hard blocker)

| context | KV per sequence, MLA (latent 512 + rope 64) |
|---|---|
| 128 K | 14 GiB |
| 1 M | 110 GiB |
| **10 M** | **1.07 TiB** |
| **20 M** | **2.1 TiB** |

A single 10 M-token request needs **more than a terabyte** of KV cache before
you store a single weight. GQA makes this worse, not better. Ring Attention
spreads the *compute*, but the cache still has to live somewhere and still has
to be read on every decode step.

10 M context is achievable in *research demonstrations* with heavy retrieval or
compression; it is not achievable as a production serving mode at this model
size. The brief's "contexte 10M tokens" and "20M tokens" lines are the part I
would drop first.

### 4.2 FP4 training with "zero loss > 0.5 % vs FP16"

FP8 training is frontier; FP4 training for a 200 T model is not validated in the
literature. Worse, the requirement is **self-contradictory as stated**: FP4 has
~1-2 decimal digits of mantissa. A model can absorb that, but "zero loss > 0.5 %"
is a claim about a specific recipe on specific data, not a property you can
assert before running the experiment. I implemented FP4/FP8 as a config knob
(`TrainConfig.quantized_linear`) precisely because it is a target, not a
guarantee.

### 4.3 Latency and throughput targets

* "Latency inference < 100 ms first token, 10 000 tokens/s": with 384 B active
  parameters, one forward pass is ~768 GFLOP of weights alone. A 100 ms TTFT
  needs ~7.7 TFLOP/s of *effective* matrix throughput per request stream, and
  filling a batch of those needs a 200 T-parameter weight footprint resident in
  memory. Doable on a dedicated Blackwell node *for a single stream*; the stated
  production throughput is a different problem.
* "Débit > 10 000 tokens/s/GPU" during training: 10 000 tokens/s × 8192 tokens
  context is 82 M token-positions/s/GPU, each needing a forward+backward pass.
  At 40 % MFU that is ~3 orders of magnitude off. This one is simply wrong.

### 4.4 The quality targets

MMLU > 95 %, MMMU > 85 %, Video-MME > 80 %, HumanEval > 95 %, MT-Bench > 9.5 —
*simultaneously* — describe a model better than any shipped system as of this
writing, on a budget of a single training run. Each one is individually
approachable; the conjunction is not something you can plan to.

## 5. The 111T variant

The second brief ("GIGA-111", 111 T total / 500B-1T active / 20 000 experts /
top-128 / 20 M context) has the same profile: the parameter count is at the
frontier edge, the active-parameter ratio makes compute manageable, and the
context and precision requirements are the impossible parts. Scaling the
architecture up does not change the KV-cache arithmetic — it makes it worse.

## 6. Energy and schedule

| claim | reality check |
|---|---|
| "500 MW" for 100 k H100 | ~70-100 MW IT load. The brief *over*-specifies power. |
| "2 GW" for 500 k B200 | ~700 MW-1 GW. Also over-specified. |
| "50 PB NVMe + 500 PB HDD" | right order for optimizer states |
| **42 months, Phases 0-10, including collecting 50 T tokens and 1 B video clips** | not credible. Collecting, cleaning and deduplicating 50 T tokens alone is a multi-year, multi-org effort; the post-training, alignment, and evaluation phases are each their own research programs. |

## 7. What this repository actually delivers

Given the above, this repository is an honest implementation of the parts that
*can* be built and tested here:

* the full architecture — GQA, MLA, RoPE/YaRN, RMSNorm+QK-Norm, SwiGLU, MoE with
  shared experts and auxiliary-loss-free balancing;
* Muon + AdamW hybrid optimizer with a warmup/WSD schedule;
* a training loop with gradient accumulation, clipping, checkpointing and exact
  resume — verified to reduce loss on CPU;
* parameter/FLOPs/memory/KV-cache calculators and a parallel-plan search that
  tells you whether a config fits *before* you burn a cluster;
* vision + audio encoders with a Perceiver resampler and a shared backbone;
* SLURM, install, and parallelism-planning scaffolding for the real cluster
  path.

What it does **not** contain, and cannot honestly claim to: a trained 200 T
model, a distributed tensor/pipeline/expert-parallel compute kernel (that is
Megatron-LM and DeepSpeed integration work), or any of the benchmark results.

If you want to move toward a real frontier model, the tractable next step is a
plan like the one in `docs/ROADMAP.md`: a 1-7 T MoE at 128 K context, validated
on a few hundred GPUs, before anything larger is attempted.
