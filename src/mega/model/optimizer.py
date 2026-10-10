"""Muon + AdamW hybrid optimiser.

Muon (Jordan et al., 2024; used by Moonshot's Kimi and reported for
DeepSeek-scale runs) applies a momentum update that is orthogonalised with a
Newton-Schulz iteration before being applied to *2-D hidden matrices*.  It is
not appropriate for embeddings, the LM head, the router, or any 1-D parameter,
so those are handled by AdamW.

The split rule is deliberately conservative:

* Muon: 2-D parameters whose name contains ``w_gate`` / ``w_up`` / ``w_down``
  (packed experts) or ``proj`` (attention / dense FFN matrices).
* AdamW: everything else (embeddings, LM head, norms, router, expert bias).

The learning rate is scaled by ``sqrt(fan_out / fan_in)`` following the Muon
reference implementation so that a single ``muon_lr`` works across shapes.
"""

from __future__ import annotations

import math

import torch

__all__ = ["MuonAdamW", "split_params"]


def _zeroth_power_via_newtonschulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Approximate the orthogonal polar factor of ``g`` with NS iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.dtype == torch.float32 else g
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.t()
        transposed = True
    x = x / (x.norm() + eps)
    for _ in range(steps):
        aa = x @ x.t()
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.t()
    return x.to(g.dtype)


def split_params(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2
        hidden = any(k in name for k in ("w_gate", "w_up", "w_down", "proj"))
        is_embed = "embed" in name or "lm_head" in name
        if is_matrix and hidden and not is_embed:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


class MuonAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 3e-4,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.muon_params, self.adamw_params = split_params(model)
        groups = [
            dict(params=self.muon_params, kind="muon", lr=muon_lr,
                 momentum=muon_momentum, ns_steps=muon_ns_steps, weight_decay=0.0),
            dict(params=self.adamw_params, kind="adamw", lr=lr, betas=betas,
                 eps=eps, weight_decay=weight_decay),
        ]
        groups = [g for g in groups if g["params"]]
        super().__init__(groups, defaults={})

    @torch.no_grad()
    def step(self, closure: callable | None = None):  # type: ignore[valid-type]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["kind"] == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    @torch.no_grad()
    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            update = g.add(buf, alpha=momentum) if momentum else g
            o = _zeroth_power_via_newtonschulz(update, group["ns_steps"])
            scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            p.add_(o, alpha=-lr * scale)

    @torch.no_grad()
    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            if wd:
                p.mul_(1 - lr * wd)
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bias1 = 1 - beta1**step
            bias2 = 1 - beta2**step
            denom = (exp_avg_sq / bias2).sqrt_().add_(eps)
            p.addcdiv_(exp_avg / bias1, denom, value=-lr)


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            list(model.parameters()),
            lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay,
        )
    return MuonAdamW(
        model,
        lr=cfg.lr,
        muon_lr=cfg.muon_lr,
        muon_momentum=cfg.muon_momentum,
        muon_ns_steps=cfg.muon_ns_steps,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def lr_at(step: int, cfg) -> float:
    """Learning-rate multiplier schedule (warmup + cosine or WSD decay)."""
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    if cfg.lr_schedule == "constant":
        return 1.0
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(progress, 1.0)
    if cfg.lr_schedule == "cosine":
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    # warmup-stable-decay: hold at 1.0 then decay linearly over the last fraction
    decay_start = 1.0 - cfg.wsd_decay_frac
    if progress <= decay_start:
        return 1.0
    tail = (progress - decay_start) / max(cfg.wsd_decay_frac, 1e-9)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 - tail)
