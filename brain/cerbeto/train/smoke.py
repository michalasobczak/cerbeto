"""Smoke training (spec 0002, gates G4 and G6): a fresh student trained for a fixed wall-clock budget."""
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

from cerbeto import paths
from cerbeto.model import checkpoint
from cerbeto.model.student import Student, StudentConfig, pick_device
from cerbeto.tokenizer import load as load_tokenizer
from cerbeto.train.data import ShelfStream

PROMPTS = ["Once upon a time", "Tom and Lily went to the park.", "The little dog was sad because"]


def student_config() -> dict:
    return yaml.safe_load((paths.CONFIG / "student.yaml").read_text())


def memory_mb(device: torch.device) -> float:
    rss = psutil.Process(os.getpid()).memory_info().rss
    gpu = torch.mps.driver_allocated_memory() if device.type == "mps" else 0
    return (rss + gpu) / 2**20


@torch.no_grad()
def evaluate(model: Student, batches: list[torch.Tensor], k: int, autocast) -> float:
    model.eval()
    with autocast():
        losses = [model.loss(b[:, :-1], b[:, 1:], k).item() for b in batches]
    model.train()
    return float(np.mean(losses))


def run(minutes: float, precision: str = "fp32", shelf: str = "s0_stories", batch_size: int = 16,
        lr: float = 2e-3, warmup: int = 200, seed: int = 0, out_root: Path = paths.REPO / "runs") -> dict:
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be fp32 or fp16 (bf16 is not allowed — spec 0002)")
    sc = student_config()
    cfg = StudentConfig.from_dict(sc["start"])
    rec = sc["recurrence"]
    device = pick_device()
    torch.manual_seed(seed)
    model = Student(cfg, seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    use_fp16 = precision == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    autocast = (lambda: torch.autocast(device.type, dtype=torch.float16)) if use_fp16 else contextlib.nullcontext

    train = ShelfStream(shelf, "train", cfg.max_seq_len)
    valid = ShelfStream(shelf, "valid", cfg.max_seq_len).fixed_batches(8, batch_size, device)
    rng = np.random.default_rng(seed)
    run_dir = out_root / f"smoke-{precision}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    log = (run_dir / "metrics.jsonl").open("w")

    def record(**row) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    initial = evaluate(model, valid, rec["k_eval"], autocast)
    record(step=0, elapsed_s=0, valid_loss=initial, params=model.num_params(), device=str(device))
    budget, start, step, tokens, peak = minutes * 60, time.time(), 0, 0, 0.0
    next_eval = 120.0
    while (elapsed := time.time() - start) < budget:
        frac = elapsed / budget
        cur_lr = lr * min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        k = int(rng.integers(rec["k_train_min"], rec["k_train_max"] + 1))
        batch = train.batch(batch_size, rng, device)
        with autocast():
            loss = model.loss(batch[:, :-1], batch[:, 1:], k)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens += batch.numel()
        if not math.isfinite(loss.item()):
            record(step=step, error="non-finite loss", loss=loss.item())
            break
        if elapsed >= next_eval:
            next_eval += 120.0
            peak = max(peak, memory_mb(device))
            record(step=step, elapsed_s=round(elapsed), train_loss=loss.item(),
                   valid_loss=evaluate(model, valid, rec["k_eval"], autocast), lr=cur_lr,
                   tokens_per_s=round(tokens / elapsed), memory_mb=round(peak))

    final = evaluate(model, valid, rec["k_eval"], autocast)
    elapsed = time.time() - start
    tok = load_tokenizer()
    samples = {}
    model.eval()
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        out = model.generate(ids, 60, k=rec["k_eval"], stop_token=tok.token_to_id("<|eos|>"))
        samples[prompt] = tok.decode(out[0].tolist())
    ckpt = checkpoint.save(model, run_dir / "checkpoint", opt, meta={"seed": seed, "run": run_dir.name})
    summary = {"precision": precision, "steps": step, "elapsed_s": round(elapsed), "initial_valid_loss": initial,
               "final_valid_loss": final, "drop": 1 - final / initial, "tokens": tokens,
               "tokens_per_s": round(tokens / elapsed), "peak_memory_mb": round(peak),
               "checkpoint": str(ckpt), "samples": samples}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    record(**{k: v for k, v in summary.items() if k != "samples"})
    for prompt, text in samples.items():
        print(f"--- {text}")
    return summary
