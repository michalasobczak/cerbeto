"""Sleep (spec 0005): the day's verified experiences are replayed into the weights.

Every step mixes the day's experiences with lessons of earlier stages (code truth) and windows of the nursery
shelves, so new skills do not push out old ones; synaptic consolidation (EWC) holds the weights that mattered for
what was already learned. A surprising experience — high loss on the correct trace — weighs more, up to ×3
(a dopamine-like signal). Loss is on the student's turn only. An interrupted sleep leaves no trace: the caller
keeps the previous version.
"""
import math
import random
from collections import Counter
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from cerbeto.curriculum import training_stream
from cerbeto.exams.school import exam_hashes
from cerbeto.memory.hippocampus import Experience
from cerbeto.model.consolidation import EWC
from cerbeto.school.smoke import IGNORE, encode_pairs, lesson_text, release
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
              memories: list[tuple[str, str, bool]] = (), validate: Callable[[], float] | None = None,
              skills_ewc: EWC | None = None) -> dict:
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
        release(device)
        val.append((step, v))
        if best is None or v > best[1]:
            best = (step, v, {n: p.detach().clone() for n, p in model.state_dict().items()})
        log(f"sleep · step {step}/{total} · validation {100 * v:.1f}% (best {100 * best[1]:.1f}% at step {best[0]})")

    losses = []
    diet = Counter()                           # tokens the night showed the student, by source (inputs, padding excluded)
    pad = tok.token_to_id("<|pad|>")
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
            sources = ["today"] * len(batch)
            if n_recall:
                batch = batch + [(p, t) for p, t, _ in rng.choices(memories, weights=recall_w, k=n_recall)]
                sources += ["earlier_days"] * n_recall
            if replay:                         # as many lessons of earlier stages, with the reference solution
                old = []
                while len(old) < cfg["batch"]:
                    t = next(replay)
                    if t.stage in earlier:
                        old.append(lesson_text(t))
                batch = batch + old
                sources += ["earlier_stages"] * len(old)
            k = int(nrng.integers(cfg.get("k_min", 1), cfg.get("k_max", 6) + 1))   # passes of the core
            x, y = encode_pairs(batch, tok, device)
            for source, n in zip(sources, (x != pad).sum(1).tolist()):
                diet[source] += n
            lesson = weighted_loss(model(x, k), y, cfg["surprise_cap"])
            windows = shelves[step % 2].batch(cfg["replay_windows"], nrng, device)
            diet[shelves[step % 2].shelf] += windows[:, :-1].numel()
            language = model.loss(windows[:, :-1], windows[:, 1:], k)
            loss = lesson + language
            if ewc is not None:
                loss = loss + cfg["ewc_lambda"] / 2 * ewc.penalty(model)
            if skills_ewc is not None:
                loss = loss + cfg["ewc_skills_lambda"] / 2 * skills_ewc.penalty(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append((lesson.item(), language.item()))
            step += 1
            if step % 25 == 0 or step == total:
                release(device)
                recent = losses[-25:]
                log(f"sleep · step {step}/{total} · lesson loss {sum(a for a, _ in recent) / len(recent):.3f} · "
                    f"language loss {sum(c for _, c in recent) / len(recent):.3f}")
            if validate and (step % every == 0 or step == total):
                check()
    model.eval()
    kept = step
    end = val[-1][1] if val and val[-1][0] == step else None
    tolerance = cfg.get("keep_tolerance", 0.0)
    if best is not None and best[0] != step and end is not None and best[1] - end <= tolerance:
        # the validation set is small: one seed alone moves a night by ±3 pp (bench, 2026-11-06), so a best check
        # within `tolerance` of the end is noise — the night's learning goes to the exam, which decides
        log(f"sleep · keeping the end of the night (validation {100 * end:.1f}%, within "
            f"{100 * tolerance:.1f} pp of the best {100 * best[1]:.1f}% at step {best[0]})")
    elif best is not None and best[0] != step:
        model.load_state_dict(best[2])
        kept = best[0]
        log(f"sleep · keeping the weights of step {kept} (validation {100 * best[1]:.1f}%)")
    return {"steps": step, "kept_step": kept, "validation": val, "experiences": len(experiences),
            "diet": dict(diet), "lesson_loss": sum(a for a, _ in losses[-25:]) / max(1, len(losses[-25:])),
            "language_loss": sum(c for _, c in losses[-25:]) / max(1, len(losses[-25:]))}


def estimate_skills_ewc(model, device, memories: list[tuple[str, str, bool]], n: int, seed: int = 0,
                        k: int = 4) -> EWC | None:
    """Importance of every weight for what the student already does right in this stage, at the current weights.

    The nursery EWC protects only language; nothing held the skills of earlier school days, so a night traded them
    (bench, 2026-11-05: agree −20, categories −12 while clarify +37). Squared gradients of the lesson loss, one
    earlier lesson at a time (a batch mean squares away almost everything: the nursery Fisher sums to 1.7 over 34.6M
    weights), normalised to mean 1 per weight, so `ewc_skills_lambda` reads on the same scale for any student size."""
    right = [(p, t) for p, t, c in memories if c]
    if not right:
        return None
    tok = load_tokenizer()
    sample = random.Random(seed).sample(right, min(n, len(right)))
    fisher = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in model.named_parameters()}
    model.eval()
    for pair in sample:
        model.zero_grad(set_to_none=True)
        x, y = encode_pairs([pair], tok, device)
        F.cross_entropy(model(x, k).float().transpose(1, 2), y, ignore_index=IGNORE).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().float() ** 2
    model.zero_grad(set_to_none=True)
    total = sum(float(v.sum()) for v in fisher.values())
    count = sum(v.numel() for v in fisher.values())
    scale = count / max(total, 1e-30)
    anchor = {name: p.detach().float().clone() for name, p in model.named_parameters()}
    return EWC({name: v * scale for name, v in fisher.items()}, anchor)


def estimate_ewc(model, device, batches: int, seed: int = 0) -> EWC:
    """Importance of every weight for the nursery language (shelf windows) at the current weights."""
    shelf = ShelfStream("s0_speech", "train", model.cfg.max_seq_len)
    windows = shelf.fixed_batches(batches, 4, device, seed)
    ewc = EWC.estimate(model, windows, k=4)
    model.eval()
    return ewc
