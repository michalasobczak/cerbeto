"""`cerbeto school smoke`: does the student learn the curriculum at all? (spec 0004, smoke check before 0005)

A short, direct lesson run without a teacher, hippocampus or sleep: lessons straight from the generators
(training seeds, never an exam prompt; one stage — preschool by default, school order), the student learns the reference reply — with its step-by-step
solution inside `<|think|> … <|/think|>` — and language replay from the nursery shelves keeps it from
forgetting how to talk. Loss is computed on the student's turn only. Afterwards: the school exams again.
It starts from a nursery checkpoint and writes a separate student (`checkpoints/smoke/`), never the original.
"""
import json
import math
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from cerbeto import paths
from cerbeto.curriculum import Task, training_stream
from cerbeto.exams.school import exam_hashes
from cerbeto.model import checkpoint
from cerbeto.talk import load_student
from cerbeto.tokenizer import load as load_tokenizer
from cerbeto.train.data import ShelfStream

IGNORE = -100


def lesson_text(task: Task) -> tuple[str, str]:
    """(prompt, target) in the chat format: the student thinks first when the task has a reference solution."""
    think = f"<|think|> {' '.join(task.steps)} <|/think|>" if task.steps and task.answer_type != "unknown" else ""
    return f"<|human|> {task.prompt}\n<|student|>", f"{think}<|answer|> {task.reply}<|eos|>"


def encode_lessons(tasks: list[Task], tok, device, max_len: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    eos, pad = tok.token_to_id("<|eos|>"), tok.token_to_id("<|pad|>")
    rows, labels = [], []
    for t in tasks:
        prompt, target = lesson_text(t)
        p, a = [eos] + tok.encode(prompt).ids, tok.encode(target).ids
        ids = (p + a)[:max_len + 1]
        lab = ([IGNORE] * len(p) + a)[:max_len + 1]
        rows.append(ids)
        labels.append(lab)
    width = max(map(len, rows))
    x = torch.full((len(rows), width), pad, dtype=torch.long)
    y = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(rows, labels)):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(lab)] = torch.tensor(lab)
    return x[:, :-1].to(device), y[:, 1:].to(device)   # predict token t+1 from tokens ≤ t


def run(student: str = "growing", version: str | None = None, minutes: float = 30, stage: str | None = "preschool",
        lessons_per_step: int = 32, replay_batch: int = 4, lr: float = 3e-4, seed: int = 0,
        device: str | None = None) -> dict:
    model, version, dev = load_student(student, version, device)
    tok = load_tokenizer()
    parent = paths.REPO / "checkpoints" / student / version
    out_root = paths.REPO / "checkpoints" / "smoke"
    run_dir = paths.REPO / "runs" / "school" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    exam = exam_hashes()
    if not exam:
        raise SystemExit("no exam pools — run `cerbeto exam build` first, so lessons can avoid exam prompts")
    lessons = training_stream(seed, stage=stage, exclude=exam)   # school order: one stage at a time
    replay = {s: ShelfStream(s, "train", model.cfg.max_seq_len) for s in ("s0_speech", "s0_stories")}
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup, budget = 100, minutes * 60
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    log = (run_dir / "metrics.jsonl").open("a")
    start, step, seen = time.time(), 0, 0
    print(f"smoke school: {student}/{version} on {dev}, stage {stage or 'all'}, {minutes} min, {lessons_per_step} lessons + "
          f"{replay_batch}×{model.cfg.max_seq_len} replay tokens per step")
    while not stop["now"] and time.time() - start < budget:
        frac = (time.time() - start) / budget
        for g in opt.param_groups:   # warm-up, then cosine to 10%
            g["lr"] = lr * min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        model.train()
        k = int(rng.integers(1, 7))
        x, y = encode_lessons([next(lessons) for _ in range(lessons_per_step)], tok, dev)
        logits = model(x, k)
        lesson_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE)
        shelf = "s0_speech" if rng.random() < 0.5 else "s0_stories"
        batch = replay[shelf].batch(replay_batch, rng, dev)
        replay_loss = model.loss(batch[:, :-1], batch[:, 1:], k)
        opt.zero_grad(set_to_none=True)
        (lesson_loss + replay_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        seen += lessons_per_step
        if step % 50 == 0:
            row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "lessons": seen,
                   "lesson_loss": round(lesson_loss.item(), 4), "replay_loss": round(replay_loss.item(), 4),
                   "lr": g["lr"], "minutes": round((time.time() - start) / 60, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row)
    n = len(list(out_root.glob("v*"))) + 1 if out_root.exists() else 1
    path = checkpoint.save(model, out_root / f"v{n:04d}", opt, parent=parent,
                           meta={"smoke": {"from": f"{student}/{version}", "stage": stage, "steps": step, "lessons": seen,
                                           "minutes": round((time.time() - start) / 60, 1), "lr": lr}})
    (out_root / "LATEST").write_text(path.name)
    print(f"saved {path} after {step} steps, {seen:,} lessons")
    return {"path": str(path), "steps": step, "lessons": seen}
