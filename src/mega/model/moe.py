"""Mixture-of-Experts layer with shared experts and load balancing.

Design notes
------------
* Routed experts are stored as three packed 3-D tensors
  (``w_gate``, ``w_up``, ``w_down``) instead of a ``ModuleList``.  This keeps
  the state dict small, makes parameter counting trivial, and is the layout
  expert-parallel sharding expects (slice dim 0 across ranks).
* Every token always passes through ``n_shared_experts`` shared experts
  (DeepSeekMoE, Dai et al. 2024) which capture common knowledge and stabilise
  routing.
* Routing supports both ``softmax`` (Switch/Mixtral style) and ``sigmoid``
  scoring.  Selected weights are renormalised to sum to one.
* Two balancing mechanisms are available and can be combined:
  - a Switch-style auxiliary load-balance loss (``aux_loss_coef``);
  - auxiliary-loss-free balancing via a learned per-expert bias that is
    updated from token counts and never receives gradients (DeepSeek-V3).

Dispatch is a Python loop over experts; each iteration only touches the
contiguous slice of tokens routed to that expert.  This is intentionally simple
and exact.  For very large expert counts a grouped GEMM / all-to-all kernel
should replace it (see ``parallel/plan.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MoEStats", "MoELayer"]


@dataclass
class MoEStats:
    aux_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    z_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    tokens_per_expert: torch.Tensor | None = None
    router_entropy: torch.Tensor | None = None

    def detached(self) -> MoEStats:
        return MoEStats(
            aux_loss=self.aux_loss.detach(),
            z_loss=self.z_loss.detach(),
            tokens_per_expert=None if self.tokens_per_expert is None else self.tokens_per_expert,
            router_entropy=None if self.router_entropy is None else self.router_entropy,
        )


class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.hidden = cfg.resolved_expert_hidden
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.moe_top_k
        self.n_shared = cfg.n_shared_experts

        self.router = nn.Linear(cfg.d_model, self.n_experts, bias=False)

        def packed():
            return nn.Parameter(torch.empty(self.n_experts, self.hidden, cfg.d_model))

        self.w_gate = packed()
        self.w_up = packed()
        self.w_down = nn.Parameter(torch.empty(self.n_experts, cfg.d_model, self.hidden))

        if self.n_shared > 0:
            self.shared = nn.ModuleList(
                [self._make_shared() for _ in range(self.n_shared)]
            )
        else:
            self.shared = None

        if cfg.router_bias:
            self.register_buffer("expert_bias", torch.zeros(self.n_experts), persistent=True)
        else:
            self.expert_bias = None

        self._reset_parameters()

    def _make_shared(self) -> nn.Module:
        from .ffn import SwiGLU

        return SwiGLU(self.d_model, self.hidden)

    def _reset_parameters(self) -> None:
        std = self.cfg.init_std
        for p in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(p, mean=0.0, std=std)
        nn.init.normal_(self.router.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------ routing
    def route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(top_idx [N,k], top_weight [N,k], logits [N,E])``."""
        logits = self.router(x_flat)
        if self.cfg.router_score == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)

        if self.training and self.cfg.router_jitter > 0:
            scores = scores * (
                1.0 + self.cfg.router_jitter * torch.randn_like(scores)
            )

        sel = scores if self.expert_bias is None else scores + self.expert_bias
        top_val, top_idx = sel.topk(self.top_k, dim=-1)
        top_weight = top_val / (top_val.sum(dim=-1, keepdim=True) + 1e-9)
        return top_idx, top_weight, logits

    def _expert_ffn(self, e: int, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(x @ self.w_gate[e].t()) * (x @ self.w_up[e].t())
        return h @ self.w_down[e].t()

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoEStats]:
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)
        n = x_flat.shape[0]

        top_idx, top_weight, logits = self.route(x_flat)
        # accumulate in the embedding dtype (fp32 outside autocast) so that
        # expert GEMMs may run in bf16 without losing the residual accumulator
        out = torch.zeros_like(x_flat)

        for e in range(self.n_experts):
            match = top_idx == e
            if not bool(match.any()):
                continue
            token_idx, slot = match.nonzero(as_tuple=True)
            xe = x_flat.index_select(0, token_idx)
            w = top_weight[token_idx, slot].unsqueeze(-1)
            contrib = (w * self._expert_ffn(e, xe)).to(out.dtype)
            out.index_add_(0, token_idx, contrib)

        if self.shared is not None:
            for expert in self.shared:
                out = out + expert(x_flat).to(out.dtype)

        stats = self._stats(logits, top_idx, n)
        self._last_tokens_per_expert = stats.tokens_per_expert.detach()
        return out.view(b, t, d), stats

    def _stats(self, logits: torch.Tensor, top_idx: torch.Tensor, n: int) -> MoEStats:
        probs = logits.softmax(dim=-1)
        one_hot = F.one_hot(top_idx, self.n_experts).to(probs.dtype).sum(dim=1)  # [N,E]
        tokens_per_expert = one_hot.sum(dim=0)
        frac = tokens_per_expert / max(n * self.top_k, 1)
        mean_prob = probs.mean(dim=0)
        aux = self.n_experts * (frac * mean_prob).sum()
        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean()
        return MoEStats(
            aux_loss=aux * self.cfg.aux_loss_coef,
            z_loss=z * self.cfg.z_loss_coef,
            tokens_per_expert=tokens_per_expert,
            router_entropy=entropy,
        )

    @torch.no_grad()
    def update_expert_bias(self, tokens_per_expert: torch.Tensor, gamma: float = 1e-3) -> None:
        """Auxiliary-loss-free balancing step (DeepSeek-V3, eq. 14-16)."""
        if self.expert_bias is None:
            return
        target = tokens_per_expert.float().mean()
        sign = torch.sign(target - tokens_per_expert.float())
        self.expert_bias.add_(gamma * sign)

    # ------------------------------------------------------------- bookkeeping
    def routed_param_count(self) -> int:
        per = 3 * self.hidden * self.d_model
        return self.n_experts * per

    def active_param_count(self) -> int:
        per = 3 * self.hidden * self.d_model
        return self.top_k * per
