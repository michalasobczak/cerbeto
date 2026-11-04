"""Sleep (spec 0005): the day's verified experiences are replayed into the weights.

Every step mixes the day's experiences with lessons of earlier stages (code truth) and windows of the nursery
shelves, so new skills do not push out old ones; synaptic consolidation (EWC) holds the weights that mattered for
what was already learned. A surprising experience — high loss on the correct trace — weighs more, up to ×3
(a dopamine-like signal). Loss is on the student's turn only. An interrupted sleep leaves no trace: the caller
keeps the previous version.
"""
import math
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from cerbeto.curriculum import training_stream
from cerbeto.exams.school import exam_hashes
from cerbeto.memory.hippocampus import Experience
from cerbeto.model.consolidation import EWC
from cerbeto.school.smoke import IGNORE, encode_pairs, lesson_text
from cerbeto.tokenizer import load as load_tokenizer
from cerbeto.train.data import ShelfStream


class Interrupted(Exception):
    pass


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, cap: float) -> torch.Tensor:
    """Mean over experiences of their own loss, each weighted by its surprise relative to the batch (1…cap)."""
    tok = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=IGNORE, reduction="none")
    mask = (labels != IGNORE).float()
    per = (tok * mask).sum(1) / mask.sum(1).clamp(min=1)
    weight = (per.detach() / per.detach().mean().clamp(min=1e-6)).clamp(1.0, cap)
    return (per * weight).sum() / weight.sum()


def run_sleep(model, device, experiences: list[Experience], earlier: list[str], cfg: dict, ewc: EWC | None,
              seed: int = 0, stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print,
              memories: list[tuple[str, str, bool]] = (), validate: Callable[[], float] | None = None) -> dict:
    tok = load_tokenizer()
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    shelves = [ShelfStream(s, "train", model.cfg.max_seq_len) for s in ("s0_speech", "s0_stories")]
    replay = training_stream(seed=20_000 + seed, stage=None, exclude=exam_hashes()) if earlier else None
    pairs = [(e.prompt, e.target) for e in experiences]
    # earlier days of this stage: mistakes come back more often (×mistake_weight)
    recall_w = [1.0 if right else cfg.get("mistake_weight", 2.0) for _, _, right in memories]
    n_recall = round(cfg["batch"] * cfg.get("memory_share", 1.0)) if memories else 0
    steps_per_epoch = math.ceil(len(pairs) / cfg["batch"])
    total = steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    every = cfg.get("validate_every", 35)
    val, best = [], None

    def check() -> None:                       # early stopping: remember the weights with the best validation
        nonlocal best
        model.eval()
        v = validate()
        model.train()
        val.append((step, v))
        if best is None or v > best[1]:
            best = (step, v, {n: p.detach().clone() for n, p in model.state_dict().items()})
        log(f"sleep · step {step}/{total} · validation {100 * v:.1f}% (best {100 * best[1]:.1f}% at step {best[0]})")

    losses = []
    step = 0
    if validate:
        check()
    model.train()
    for epoch in range(cfg["epochs"]):
        rng.shuffle(pairs)
        for b in range(0, len(pairs), cfg["batch"]):
            if stop():
                raise Interrupted
            for g in opt.param_groups:         # warm-up, then cosine to 10%
                g["lr"] = cfg["lr"] * min(1.0, (step + 1) / cfg["warmup"]) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
            batch = pairs[b:b + cfg["batch"]]
            if n_recall:
                batch = batch + [(p, t) for p, t, _ in rng.choices(memories, weights=recall_w, k=n_recall)]
            if replay:                         # as many lessons of earlier stages, with the reference solution
                old = []
                while len(old) < cfg["batch"]:
                    t = next(replay)
                    if t.stage in earlier:
                        old.append(lesson_text(t))
                batch = batch + old
            k = int(nrng.integers(1, 7))
            x, y = encode_pairs(batch, tok, device)
            lesson = weighted_loss(model(x, k), y, cfg["surprise_cap"])
            windows = shelves[step % 2].batch(cfg["replay_windows"], nrng, device)
            language = model.loss(windows[:, :-1], windows[:, 1:], k)
            loss = lesson + language
            if ewc is not None:
                loss = loss + cfg["ewc_lambda"] / 2 * ewc.penalty(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append((lesson.item(), language.item()))
            step += 1
            if step % 25 == 0 or step == total:
                recent = losses[-25:]
                log(f"sleep · step {step}/{total} · lesson loss {sum(a for a, _ in recent) / len(recent):.3f} · "
                    f"language loss {sum(c for _, c in recent) / len(recent):.3f}")
            if validate and (step % every == 0 or step == total):
                check()
    model.eval()
    kept = step
    if best is not None and best[0] != step:
        model.load_state_dict(best[2])
        kept = best[0]
        log(f"sleep · keeping the weights of step {kept} (validation {100 * best[1]:.1f}%)")
    return {"steps": step, "kept_step": kept, "validation": val, "experiences": len(experiences),
            "lesson_loss": sum(a for a, _ in losses[-25:]) / max(1, len(losses[-25:])),
            "language_loss": sum(c for _, c in losses[-25:]) / max(1, len(losses[-25:]))}


def estimate_ewc(model, device, batches: int, seed: int = 0) -> EWC:
    """Importance of every weight for the nursery language (shelf windows) at the current weights."""
    shelf = ShelfStream("s0_speech", "train", model.cfg.max_seq_len)
    windows = shelf.fixed_batches(batches, 4, device, seed)
    ewc = EWC.estimate(model, windows, k=4)
    model.eval()
    return ewc
