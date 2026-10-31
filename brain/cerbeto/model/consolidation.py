"""Synaptic consolidation — Elastic Weight Consolidation (Kirkpatrick et al., 2017).

After a grade is passed we store each weight's importance (diagonal Fisher information) and its value.
During later sleeps the penalty  λ/2 · Σ F · (θ − θ*)²  makes important weights resistant to change.
Weights added by growth after consolidation are unconstrained (their Fisher slice is zero).
"""
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from cerbeto.model.student import Student


class EWC:
    def __init__(self, fisher: dict[str, torch.Tensor], anchor: dict[str, torch.Tensor]):
        self.fisher, self.anchor = fisher, anchor

    @classmethod
    def estimate(cls, model: Student, batches: Iterable[torch.Tensor], k: int) -> "EWC":
        """Empirical diagonal Fisher: mean squared gradient of the LM loss over `batches` of token windows."""
        fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters()}
        model.eval()
        n = 0
        for batch in batches:
            model.zero_grad(set_to_none=True)
            model.loss(batch[:, :-1], batch[:, 1:], k).backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach().float() ** 2
            n += 1
        model.zero_grad(set_to_none=True)
        fisher = {k_: v / max(n, 1) for k_, v in fisher.items()}
        anchor = {n_: p.detach().float().clone() for n_, p in model.named_parameters()}
        return cls(fisher, anchor)

    def penalty(self, model: Student) -> torch.Tensor:
        total = None
        for name, p in model.named_parameters():
            if name not in self.fisher:
                continue
            f, a = self.fisher[name], self.anchor[name]
            region = tuple(slice(0, min(x, y)) for x, y in zip(f.shape, p.shape))
            term = (f[region] * (p[region].float() - a[region]) ** 2).sum()
            total = term if total is None else total + term
        return total if total is not None else torch.zeros((), device=next(model.parameters()).device)

    def save(self, path: Path) -> None:
        tensors = {f"fisher.{k}": v.cpu().contiguous() for k, v in self.fisher.items()}
        tensors |= {f"anchor.{k}": v.cpu().contiguous() for k, v in self.anchor.items()}
        save_file(tensors, str(path))

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> "EWC":
        t = load_file(str(path), device=str(device))
        return cls({k[7:]: v for k, v in t.items() if k.startswith("fisher.")},
                   {k[7:]: v for k, v in t.items() if k.startswith("anchor.")})
