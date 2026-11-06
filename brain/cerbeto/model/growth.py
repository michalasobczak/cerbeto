"""Neurogenesis, pruning and neuron renewal for the student.

Growth is function-preserving: every new unit has a *zero writer side* (the weights that put its
output into the residual stream or the logits) and a *small random reader side*. Before training the
grown network computes exactly what the old one did, but gradients reach the new units at once, so
they can start learning (a zero/zero init would be a dead saddle point).

Every operator works on a deep copy and returns `(new_model, event)`. `apply` checks the hard
parameter limit and the |Δ loss| guard and appends the event to the growth log.
"""
import copy
import inspect
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from cerbeto.model.student import Block, Student, count_params


class GrowthRejected(RuntimeError):
    """Raised when an operation would break the parameter limit or change the student's behaviour."""


def _resized(weight: torch.Tensor, out_f: int, in_f: int, new_rows: str, new_cols: str, std: float) -> nn.Parameter:
    """Copy `weight` into a larger (out_f, in_f) matrix; fill new rows/cols with 'zero' or 'random'."""
    old_out, old_in = weight.shape
    w = torch.zeros(out_f, in_f, device=weight.device, dtype=weight.dtype)
    if new_rows == "random":
        w[old_out:].normal_(0, std)
    if new_cols == "random":
        w[:, old_in:].normal_(0, std)
    w[:old_out, :old_in] = weight.detach()
    return nn.Parameter(w)


def _linear(lin: nn.Linear, out_f: int, in_f: int, new_rows: str, new_cols: str, std: float) -> None:
    lin.weight = _resized(lin.weight, out_f, in_f, new_rows, new_cols, std)
    lin.out_features, lin.in_features = out_f, in_f


def _event(op: str, before: Student, after: Student, **details) -> dict:
    return {"op": op, "params_before": before.num_params(), "params_after": after.num_params(), **details}


def grow_depth(model: Student, std: float = 0.01) -> tuple[Student, dict]:
    """Add one block at the end of the recurrent core; its attention and MLP outputs start at zero."""
    new = copy.deepcopy(model)
    device = next(model.parameters()).device
    block = Block(model.cfg).to(device)
    for lin in (block.attn.q, block.attn.k, block.attn.v, block.mlp.gate, block.mlp.up):
        nn.init.normal_(lin.weight, std=model.cfg.init_std)
    nn.init.zeros_(block.attn.o.weight)
    nn.init.zeros_(block.mlp.down.weight)
    new.core.append(block)
    new.cfg = model.cfg.with_(n_core=model.cfg.n_core + 1)
    return new, _event("grow_depth", model, new, where=f"core.{len(new.core) - 1}")


def unroll_core(model: Student, times: int, noise: float = 0.0) -> tuple[Student, dict]:
    """Untie the recurrent core: its blocks run `times` times become `times` × n_core separate blocks with the same
    weights, so the student at K = 1 computes exactly what it computed at K = `times`; afterwards the copies train
    apart (a plain deep stack, like Llama or Qwen).

    Exact copies barely move apart: after 18M tokens the 283M student's copies still had the same weight norms
    to 0.5% (2026-11-06). `noise` adds Gaussian noise of that fraction of each weight's spread to every copy after
    the first, so they start different; the Δ loss guard keeps the behaviour."""
    new = copy.deepcopy(model)
    new.core = nn.ModuleList(copy.deepcopy(b) for _ in range(times) for b in model.core)
    if noise:
        with torch.no_grad():
            for block in list(new.core)[model.cfg.n_core:]:
                for p in block.parameters():
                    if p.dim() > 1:
                        p.add_(torch.randn_like(p) * noise * p.std())
    new.cfg = model.cfg.with_(n_core=model.cfg.n_core * times)
    return new, _event("unroll_core", model, new, times=times, noise=noise)


def renew_attention(model: Student, where: list[int], std: float = 0.02) -> tuple[Student, dict]:
    """Give the attention of core blocks `where` a fresh start: q, k, v random, the output o zero (the block's
    attention says nothing until it learns). For a dead attention: in `v0017` core 3 has q and k ~3.5× smaller than
    the other blocks and spreads its attention evenly over every token in every pass (2026-11-06), and the 283M
    student kept four such copies. Not exactly function-preserving (the block's averaging is gone): the Δ loss guard
    decides."""
    new = copy.deepcopy(model)
    for i in where:
        attn = new.core[i].attn
        for lin in (attn.q, attn.k, attn.v):
            nn.init.normal_(lin.weight, std=std)
        nn.init.zeros_(attn.o.weight)
    return new, _event("renew_attention", model, new, where=[f"core.{i}" for i in where])


def grow_width(model: Student, extra_mlp: int = 0, extra_heads: int = 0, std: float = 0.01) -> tuple[Student, dict]:
    """Add MLP neurons and/or attention heads to every block (readers random, writers zero)."""
    new = copy.deepcopy(model)
    cfg = model.cfg
    f2, h2 = cfg.mlp_hidden + extra_mlp, cfg.n_heads + extra_heads
    inner2 = h2 * cfg.head_dim
    for block in new.blocks():
        if extra_mlp:
            _linear(block.mlp.gate, f2, cfg.d_model, "random", "zero", std)
            _linear(block.mlp.up, f2, cfg.d_model, "random", "zero", std)
            _linear(block.mlp.down, cfg.d_model, f2, "zero", "zero", std)
        if extra_heads:
            for lin in (block.attn.q, block.attn.k, block.attn.v):
                _linear(lin, inner2, cfg.d_model, "random", "zero", std)
            _linear(block.attn.o, cfg.d_model, inner2, "zero", "zero", std)
            block.attn.n_heads = h2
    new.cfg = cfg.with_(mlp_hidden=f2, n_heads=h2)
    return new, _event("grow_width", model, new, extra_mlp=extra_mlp, extra_heads=extra_heads)


def grow_model_dim(model: Student, new_d: int, std: float = 0.01) -> tuple[Student, dict]:
    """Widen the residual stream. New dimensions start at zero everywhere they are written. The RMS now
    averages over more (zero) dimensions, so every RMSNorm gain is scaled by sqrt(d_old/d_new) and the
    norm epsilon by d_old/d_new — together this keeps the normalized old dimensions exactly unchanged."""
    new = copy.deepcopy(model)
    cfg = model.cfg
    d, inner = cfg.d_model, cfg.n_heads * cfg.head_dim
    scale = math.sqrt(d / new_d)

    def norm(n) -> None:
        n.eps = cfg.norm_eps * d / new_d
        g = torch.full((new_d,), scale, device=n.weight.device, dtype=n.weight.dtype)
        g[:d] = n.weight.detach() * scale
        n.weight = nn.Parameter(g)

    emb = torch.zeros(cfg.vocab_size, new_d, device=new.embed.weight.device, dtype=new.embed.weight.dtype)
    emb[:, :d] = new.embed.weight.detach()
    new.embed.weight = nn.Parameter(emb)
    new.embed.embedding_dim = new_d
    for block in new.blocks():
        norm(block.attn_norm)
        norm(block.mlp_norm)
        for lin in (block.attn.q, block.attn.k, block.attn.v):
            _linear(lin, inner, new_d, "zero", "random", std)
        _linear(block.attn.o, new_d, inner, "zero", "zero", std)
        _linear(block.mlp.gate, cfg.mlp_hidden, new_d, "zero", "random", std)
        _linear(block.mlp.up, cfg.mlp_hidden, new_d, "zero", "random", std)
        _linear(block.mlp.down, new_d, cfg.mlp_hidden, "zero", "zero", std)
    norm(new.norm)
    new.cfg = cfg.with_(d_model=new_d, norm_eps=cfg.norm_eps * d / new_d)
    return new, _event("grow_model_dim", model, new, d_from=d, d_to=new_d)


def prune_mlp(model: Student, keep: dict[int, torch.Tensor]) -> tuple[Student, dict]:
    """Remove MLP neurons. `keep[block_index]` lists surviving neuron indices; every block keeps the same
    count so all blocks share one `mlp_hidden`. Not exactly function-preserving — `apply` guards Δ loss."""
    sizes = {len(v) for v in keep.values()}
    if len(sizes) != 1:
        raise ValueError("every block must keep the same number of MLP neurons")
    f2 = sizes.pop()
    new = copy.deepcopy(model)
    for i, block in enumerate(new.blocks()):
        idx = keep[i].to(block.mlp.gate.weight.device)
        block.mlp.gate.weight = nn.Parameter(block.mlp.gate.weight.detach()[idx].clone())
        block.mlp.up.weight = nn.Parameter(block.mlp.up.weight.detach()[idx].clone())
        block.mlp.down.weight = nn.Parameter(block.mlp.down.weight.detach()[:, idx].clone())
        block.mlp.gate.out_features = block.mlp.up.out_features = block.mlp.down.in_features = f2
    new.cfg = model.cfg.with_(mlp_hidden=f2)
    removed = model.cfg.mlp_hidden - f2
    return new, _event("prune_mlp", model, new, removed_per_block=removed)


@torch.no_grad()
def behaviour_delta(old: Student, new: Student, probe: torch.Tensor, k: int,
                    k_new: int | None = None) -> tuple[float, float]:
    """Max |Δ logits| and |Δ loss| between two students on a probe batch of token windows (`k_new`: passes of the
    new student's core, when they differ, as after `unroll_core`)."""
    old.eval(), new.eval()
    x, y = probe[:, :-1], probe[:, 1:]
    kn = k if k_new is None else k_new
    lo, ln = old(x, k).float(), new(x, kn).float()
    dl = abs(old.loss(x, y, k).item() - new.loss(x, y, kn).item())
    return (lo - ln).abs().max().item(), dl


def apply(model: Student, op: str, probe: torch.Tensor, *, k: int, max_params: int, max_delta_loss: float,
          log_path: Path | None = None, reason: str = "", k_new: int | None = None, **kwargs) -> Student:
    """Run a growth/pruning operator with the parameter limit and the |Δ loss| guard; log the outcome."""
    ops = {"grow_depth": grow_depth, "grow_width": grow_width, "grow_model_dim": grow_model_dim, "prune_mlp": prune_mlp,
           "unroll_core": unroll_core, "renew_attention": renew_attention}
    new, event = ops[op](model, **kwargs)
    event.update(reason=reason, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if count_params(new.cfg) != new.num_params():
        raise AssertionError(f"parameter formula mismatch after {op}")
    try:
        if new.num_params() > max_params:
            raise GrowthRejected(f"{op} would reach {new.num_params():,} > limit {max_params:,} parameters")
        dlogits, dloss = behaviour_delta(model, new, probe, k, k_new)
        event.update(max_delta_logits=dlogits, delta_loss=dloss)
        if dloss > max_delta_loss:
            raise GrowthRejected(f"{op} changed loss by {dloss:.4f} > {max_delta_loss}")
        event["accepted"] = True
        return new
    except GrowthRejected as e:
        event.update(accepted=False, error=str(e))
        raise
    finally:
        if log_path is not None:
            with log_path.open("a") as f:
                f.write(json.dumps(event) + "\n")


def adamw_kwargs(group: dict) -> dict:
    """Constructor arguments recoverable from an AdamW param group (groups also hold internal keys)."""
    accepted = inspect.signature(torch.optim.AdamW.__init__).parameters
    return {k: v for k, v in group.items() if k in accepted and k != "params"}


def transplant_optimizer(old_opt: torch.optim.Optimizer, old: Student, new: Student,
                         kept_index: dict[str, tuple[int, torch.Tensor]] | None = None) -> torch.optim.AdamW:
    """New AdamW for `new` that keeps the moments of every surviving weight. Grown tensors get zero
    moments in their new slices; pruned tensors keep the moments of the kept indices."""
    kept_index = kept_index or {}
    opt = torch.optim.AdamW(new.parameters(), **adamw_kwargs(old_opt.param_groups[0]))
    old_params = dict(old.named_parameters())
    for name, p in new.named_parameters():
        state = old_opt.state.get(old_params.get(name))
        if not state:
            continue
        moved = {"step": state["step"].clone()}
        for key in ("exp_avg", "exp_avg_sq"):
            src = state[key]
            if name in kept_index:
                dim, idx = kept_index[name]
                moved[key] = src.index_select(dim, idx.to(src.device)).clone()
            else:
                dst = torch.zeros_like(p)
                region = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, p.shape))
                dst[region] = src[region]
                moved[key] = dst
        opt.state[p] = moved
    return opt


class NeuronTracker:
    """Per-neuron utility (continual backprop): EMA of |hidden activation| × ‖outgoing weights‖,
    plus each neuron's age. Used for renewal (every sleep) and pruning (persistently useless neurons)."""

    def __init__(self, model: Student, decay: float = 0.99):
        self.model, self.decay = model, decay
        f = model.cfg.mlp_hidden
        device = next(model.parameters()).device
        n = model.cfg.n_blocks
        self.mean_abs = [torch.zeros(f, device=device) for _ in range(n)]
        self.age = [torch.zeros(f, dtype=torch.long, device=device) for _ in range(n)]
        self.low_count = [torch.zeros(f, dtype=torch.long, device=device) for _ in range(n)]
        self._hooks = [b.mlp.down.register_forward_pre_hook(self._hook(i)) for i, b in enumerate(model.blocks())]

    def _hook(self, i: int):
        def hook(_module, inputs):
            if self.model.training:
                a = inputs[0].detach().abs().float().mean(dim=(0, 1))
                self.mean_abs[i].mul_(self.decay).add_(a, alpha=1 - self.decay)
        return hook

    def step(self) -> None:
        for age in self.age:
            age += 1

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()

    def utilities(self) -> list[torch.Tensor]:
        return [m * b.mlp.down.weight.detach().float().norm(dim=0)
                for m, b in zip(self.mean_abs, self.model.blocks())]

    @torch.no_grad()
    def renew(self, fraction: float, maturity: int, std: float, optimizer: torch.optim.Optimizer | None = None) -> int:
        """Re-initialize the least useful mature neurons (≤ `fraction` of all): incoming weights redrawn,
        outgoing weights zeroed. The fresh neuron does not disturb the network; the only change in
        behaviour is the loss of the old, least useful contribution."""
        utils = self.utilities()
        candidates = [(u.item(), b, i) for b, (ut, age) in enumerate(zip(utils, self.age))
                      for i, u in enumerate(ut) if age[i] >= maturity]
        budget = int(fraction * sum(len(u) for u in utils))
        chosen = sorted(candidates)[:budget]
        blocks = list(self.model.blocks())
        for _, b, i in chosen:
            mlp = blocks[b].mlp
            mlp.gate.weight[i].normal_(0, std)
            mlp.up.weight[i].normal_(0, std)
            mlp.down.weight[:, i] = 0
            self.mean_abs[b][i] = 0
            self.age[b][i] = 0
            if optimizer is not None:
                for p, sl in ((mlp.gate.weight, (i,)), (mlp.up.weight, (i,)), (mlp.down.weight, (slice(None), i))):
                    st = optimizer.state.get(p)
                    if st:
                        st["exp_avg"][sl] = 0
                        st["exp_avg_sq"][sl] = 0
        return len(chosen)

    def mark_sleep(self, threshold: float) -> None:
        """Called once per sleep: count consecutive sleeps in which a neuron stayed below `threshold`."""
        for lc, u in zip(self.low_count, self.utilities()):
            low = u < threshold
            lc[low] += 1
            lc[~low] = 0

    def prune_plan(self, sleeps: int = 3) -> dict[int, torch.Tensor] | None:
        """Neurons useless for `sleeps` sleeps in a row. Blocks must stay equal-sized, so the same number
        is removed from every block (the minimum over blocks)."""
        doomed = [torch.nonzero(lc >= sleeps).flatten() for lc in self.low_count]
        n = min(len(d) for d in doomed)
        if n == 0:
            return None
        f = self.model.cfg.mlp_hidden
        plan = {}
        for b, (d, u) in enumerate(zip(doomed, self.utilities())):
            drop = d[torch.argsort(u[d])[:n]]
            mask = torch.ones(f, dtype=torch.bool, device=u.device)
            mask[drop] = False
            plan[b] = torch.nonzero(mask).flatten()
        return plan
