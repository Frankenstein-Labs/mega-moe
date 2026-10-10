# Architecture

## Shape

Decoder-only transformer. A block is pre-norm residual:

```
x = x + Attn(RMSNorm(x))
x = x + FFN(RMSNorm(x))        # FFN is SwiGLU or MoE
```

Layers where `(layer_index + 1) % moe_every == 0` use MoE; the rest use a dense
SwiGLU. Mixing them keeps the early layers cheap and the deep layers sparse.

## Attention

Two interchangeable implementations, same forward signature
`(x, cos, sin, cache) -> (out, cache)`:

### GQA
`n_heads` query heads share `n_kv_heads` key/value heads. The KV cache is
reduced by `n_heads / n_kv_heads`. QK-Norm (RMSNorm over `head_dim`) is applied
to queries and keys before RoPE, which is what keeps attention logits in range
at large scale.

### MLA (DeepSeek-V2/V3)
Keys and values come from one low-rank latent `c_kv` of width `kv_lora_rank`.
A small `qk_rope_head_dim` slice carries position and is decoupled from the rest
of the key. The cache stores `(c_kv, k_rope)` only:

```
per token bytes: (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
               = (512 + 64) * 2 = 1152 bytes   at bf16
```

versus `2 * n_kv_heads * head_dim * 2 = 4096` bytes for the GQA config it
replaces. This is the single biggest lever on long-context serving cost.

## Position

RoPE with a frequency cache built once per model and stored as a
**non-persistent** buffer — checkpoints stay small and context extension is a
config change, not a conversion step. `rope_scaling` supports:

* `linear` — divide all frequencies by `factor`;
* `yarn` — NTK-by-parts: high frequencies untouched, low frequencies divided,
  blended with a linear ramp between the `beta_slow`/`beta_fast` correction
  dimensions, plus the `mscale` attention-temperature compensation.

The rotation uses the half-split (LLaMA) convention, which preserves per-head
norm exactly — there is a test for this.

## MoE

```
logits  = router(x)                       # [N, n_experts]
scores  = softmax(logits) | sigmoid(logits)
sel     = scores + expert_bias            # bias is a buffer, no gradient
top_idx, top_val = sel.topk(top_k)
weights = top_val / top_val.sum()         # renormalised
out     = sum_e  weight * ExpertTopK_e(x) + sum_s SharedExpert_s(x)
```

* **Packed expert weights.** `w_gate`/`w_up`/`w_down` are 3-D parameters of
  shape `[n_experts, hidden, d_model]` (and the transpose for down). Small state
  dict, trivial parameter counting, and the natural layout for expert-parallel
  sharding along dim 0.
* **Shared experts** (DeepSeekMoE) always run and absorb common knowledge.
* **Balancing.** A Switch-style auxiliary loss and a router z-loss are added to
  the LM loss. `router_bias=True` additionally enables auxiliary-loss-free
  balancing: a per-expert bias is nudged by `gamma * sign(mean_count - count)`
  after each step and never receives gradients.
* **Dispatch** is an exact Python loop over experts. Each iteration touches only
  the contiguous token slice routed to that expert. Correct and testable; a
  production run needs a grouped GEMM / all-to-all kernel.

### Parameter arithmetic

For a packed expert layer, the total is `n_experts * 3 * d_model * hidden` and
the active count is `(top_k + n_shared) * 3 * d_model * hidden`. The
`MegaModel.num_parameters(active=True)` and `parallel.plan.active_params`
implementations agree exactly — asserted in `tests/test_model.py`.

## Optimizer — Muon + AdamW

| group | parameters | update |
|---|---|---|
| Muon | 2-D matrices named `*proj*`, `w_gate`, `w_up`, `w_down` | momentum, then Newton-Schulz orthogonalisation |
| AdamW | embeddings, LM head, norms, router, biases | standard AdamW |

Muon's orthogonalised update is only meaningful for 2-D hidden matrices, so the
split rule is enforced in code and tested: no embedding, router or LM-head
parameter may end up in the Muon group. The learning rate is scaled by
`sqrt(fan_out / fan_in)` so one `muon_lr` works across shapes.

Schedule: linear warmup then either cosine or warmup-stable-decay (WSD).

## Multimodal fusion

```
image/video --ViT(+temporal pool)--> features --Perceiver resampler--> N soft tokens
audio       --log-mel + conv + Tx--> features --Perceiver resampler--> N soft tokens
                                                  |
                     [soft tokens] ++ [text embeddings] -> shared backbone -> LM head
```

The resampler gives a fixed token budget regardless of resolution or clip
length. One set of transformer weights, one next-token-prediction loss, modality
embeddings so the backbone can tell soft tokens from text.

`MultimodalModel` **prepends** the soft tokens. Production models use placeholder
token ids and `masked_scatter` to interleave them anywhere in the sequence; the
cross-attention path is identical, only the sequence plumbing is simplified.

## What happens at scale

The reference implementation is exact but not parallel. Moving to a real run
means replacing, not extending, three things:

1. **MoE dispatch** → all-to-all + grouped GEMM (expert parallelism along the
   expert dimension).
2. **Attention** → ring attention / context parallelism for the long-context
   phase (the `torch SDPA` path already selects FlashAttention on GPU).
3. **Optimizer sharding** → ZeRO-1/2/3 or FSDP for the 3.2 PB of state.

`parallel/plan.py` computes the target topology; `train/distributed.py` validates
that the config's topology matches the real world size before any job starts.
