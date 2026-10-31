"""Nursery training loop (spec 0003): phased data mix, gradient accumulation, growth on plateau,
versioned checkpoints and exact resumption after an interruption."""
import json
import math
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cerbeto import paths
from cerbeto.model import checkpoint, growth
from cerbeto.model.student import Student, StudentConfig, pick_device
from cerbeto.train.data import ShelfStream


def load_config(name: str = "nursery") -> dict:
    return yaml.safe_load((paths.CONFIG / f"{name}.yaml").read_text())


def student_yaml() -> dict:
    return yaml.safe_load((paths.CONFIG / "student.yaml").read_text())


class Trainer:
    def __init__(self, student: str, cfg: dict, shelves_root: Path = paths.SHELVES,
                 out_root: Path = paths.REPO / "checkpoints", run_root: Path = paths.REPO / "runs" / "nursery",
                 device: torch.device | None = None, base_config: StudentConfig | None = None):
        self.name, self.cfg = student, cfg
        self.device = device or pick_device()
        sy = student_yaml()
        self.k_min, self.k_max = sy["recurrence"]["k_train_min"], sy["recurrence"]["k_train_max"]
        self.k_eval, self.max_params = sy["recurrence"]["k_eval"], sy["max_params"]
        self.max_delta_loss, self.new_std = sy["growth"]["max_delta_loss"], sy["growth"]["new_weight_std"]
        self.ckpt_dir, self.run_dir = out_root / student, run_root / student
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.growth_log = self.run_dir / "growth_log.jsonl"
        self.seq_len = (base_config or StudentConfig.from_dict(sy["start"])).max_seq_len

        shelves = {s for ph in cfg["phases"] for s in ph["mix"]}
        self.train = {s: ShelfStream(s, "train", self.seq_len, shelves_root) for s in shelves}
        self.valid = {s: ShelfStream(s, "valid", self.seq_len, shelves_root)
                      .fixed_batches(cfg["eval_batches"], cfg["micro_batch"], self.device, seed=1) for s in shelves}
        self.probe = next(iter(self.valid.values()))[0]

        base = base_config or StudentConfig.from_dict(sy["start"])
        spec = cfg["students"][student]
        self.grows = spec["grow"]
        self.model = Student(base.with_(**spec.get("config", {})), seed=cfg["seed"]).to(self.device)
        self.opt = self._optimizer(self.model)
        self.rng = np.random.default_rng(cfg["seed"])
        self.state = {"step": 0, "tokens": 0, "growth_index": 0, "evals": [], "last_growth_tokens": 0,
                      "version": 0, "segment": {"start_step": 0, "start_tokens": 0, "peak": cfg["lr"],
                                                "warmup": cfg["warmup_steps"]}}
        self.model.checkpointing = cfg["activation_checkpointing"]

    def _optimizer(self, model: Student) -> torch.optim.AdamW:
        return torch.optim.AdamW(model.parameters(), lr=self.cfg["lr"], betas=(0.9, 0.95),
                                 weight_decay=self.cfg["weight_decay"])

    # --- schedule -------------------------------------------------------------------------------
    def phase(self) -> dict:
        done = self.state["tokens"] / self.cfg["total_tokens"]
        acc = 0.0
        for ph in self.cfg["phases"]:
            acc += ph["fraction"]
            if done < acc:
                return ph
        return self.cfg["phases"][-1]

    def lr(self) -> float:
        """Warm-up + cosine within the current segment. Every growth starts a new segment (re-warm to
        `rewarm_peak_ratio` · lr, then decay to the floor by the end of training) so new neurons can learn."""
        seg = self.state["segment"]
        warm = min(1.0, (self.state["step"] - seg["start_step"] + 1) / seg["warmup"])
        span = max(1, self.cfg["total_tokens"] - seg["start_tokens"])
        frac = min(1.0, (self.state["tokens"] - seg["start_tokens"]) / span)
        floor = self.cfg["lr"] * self.cfg["min_lr_ratio"]
        return warm * (floor + (seg["peak"] - floor) * 0.5 * (1 + math.cos(math.pi * frac)))

    def next_batch(self) -> tuple[torch.Tensor, int]:
        mix = self.phase()["mix"]
        names, weights = list(mix), np.array(list(mix.values()), dtype=float)
        shelf = names[self.rng.choice(len(names), p=weights / weights.sum())]
        k = int(self.rng.integers(self.k_min, self.k_max + 1))
        return self.train[shelf].batch(self.cfg["micro_batch"], self.rng, self.device), k

    # --- training -------------------------------------------------------------------------------
    def step(self) -> float:
        for g in self.opt.param_groups:
            g["lr"] = self.lr()
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(self.cfg["accumulation"]):
            batch, k = self.next_batch()
            loss = self.model.loss(batch[:, :-1], batch[:, 1:], k) / self.cfg["accumulation"]
            loss.backward()
            total += loss.item()
            self.state["tokens"] += batch[:, 1:].numel()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
        self.opt.step()
        self.state["step"] += 1
        return total

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        out = {s: float(np.mean([self.model.loss(b[:, :-1], b[:, 1:], self.k_eval).item() for b in bs]))
               for s, bs in self.valid.items()}
        out["mean"] = float(np.mean(list(out.values())))
        return out

    def deadline_reached(self) -> bool:
        deadlines = self.cfg["growth"].get("deadlines_tokens", [])
        i = self.state["growth_index"]
        return i < len(deadlines) and self.state["tokens"] >= deadlines[i]

    def plateau(self) -> bool:
        g = self.cfg["growth"]
        losses = [e["mean"] for e in self.state["evals"]]
        w = g["plateau_windows"]
        if len(losses) <= w or self.state["tokens"] - self.state["last_growth_tokens"] < g["min_tokens_between"]:
            return False
        return (losses[-w - 1] - losses[-1]) / losses[-w - 1] < g["min_improvement"]

    def grow(self, trigger: str = "plateau") -> None:
        schedule = self.cfg["growth"]["schedule"]
        if self.state["growth_index"] >= len(schedule):
            return
        old, model = self.model, self.model
        try:
            for op in schedule[self.state["growth_index"]]:
                kwargs = {k: v for k, v in op.items() if k != "op"}
                if op["op"] != "prune_mlp":
                    kwargs["std"] = self.new_std
                model = growth.apply(model, op["op"], self.probe, k=self.k_eval, max_params=self.max_params,
                                     max_delta_loss=self.max_delta_loss, log_path=self.growth_log,
                                     reason=f"{trigger} at {self.state['tokens']:,} tokens", **kwargs)
        except growth.GrowthRejected as e:
            self.log(event="growth_rejected", error=str(e))
            self.state["last_growth_tokens"] = self.state["tokens"]
            return
        model.checkpointing = old.checkpointing
        self.opt = growth.transplant_optimizer(self.opt, old, model)
        self.model = model
        self.state["growth_index"] += 1
        self.state["last_growth_tokens"] = self.state["tokens"]
        self.state["segment"] = {"start_step": self.state["step"], "start_tokens": self.state["tokens"],
                                 "peak": self.cfg["lr"] * self.cfg["rewarm_peak_ratio"],
                                 "warmup": self.cfg["rewarm_steps"]}
        self.log(event="growth", trigger=trigger, params=model.num_params(), config=model.cfg.to_dict())

    # --- persistence ----------------------------------------------------------------------------
    def log(self, **row) -> None:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": self.state["step"],
               "tokens": self.state["tokens"], **row}
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def save(self) -> Path:
        self.state["version"] += 1
        parent = self.latest()
        out = checkpoint.save(self.model, self.ckpt_dir / f"v{self.state['version']:04d}", self.opt, parent=parent,
                              meta={"student": self.name, "stage": 0, "tokens": self.state["tokens"]})
        if self.growth_log.exists():
            shutil.copy(self.growth_log, out / "growth_log.jsonl")
        state = {**self.state, "rng": self.rng.bit_generator.state}
        (out / "training_state.json").write_text(json.dumps(state))
        (self.ckpt_dir / "LATEST").write_text(out.name)
        return out

    def latest(self) -> Path | None:
        marker = self.ckpt_dir / "LATEST"
        return self.ckpt_dir / marker.read_text().strip() if marker.exists() else None

    def resume(self) -> bool:
        path = self.latest()
        if path is None:
            return False
        self.model, self.opt = checkpoint.load(path, self.device, with_optimizer=True)
        self.model.checkpointing = self.cfg["activation_checkpointing"]
        state = json.loads((path / "training_state.json").read_text())
        self.rng.bit_generator.state = state.pop("rng")
        self.state = state
        return True

    # --- main loop ------------------------------------------------------------------------------
    def run(self, max_tokens: int | None = None) -> None:
        stop = {"now": False}

        def handler(*_):
            stop["now"] = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        target = min(max_tokens or self.cfg["total_tokens"], self.cfg["total_tokens"])
        last_save, next_eval = time.time(), self.state["tokens"] + self.cfg["eval_every_tokens"]
        t0, tok0 = time.time(), self.state["tokens"]
        if not self.state["evals"]:
            ev = self.evaluate()
            self.state["evals"].append(ev)
            self.log(event="eval", valid=ev, params=self.model.num_params(), phase=self.phase()["name"])
        while self.state["tokens"] < target and not stop["now"]:
            loss = self.step()
            if self.state["tokens"] >= next_eval:
                next_eval += self.cfg["eval_every_tokens"]
                ev = self.evaluate()
                self.state["evals"].append(ev)
                rate = (self.state["tokens"] - tok0) / (time.time() - t0)
                self.log(event="eval", train_loss=loss, valid=ev, lr=self.lr(), params=self.model.num_params(),
                         phase=self.phase()["name"], tokens_per_s=round(rate))
                if self.grows and self.deadline_reached():
                    self.grow("deadline")
                elif self.grows and self.plateau():
                    self.grow("plateau")
            if time.time() - last_save >= self.cfg["checkpoint_every_minutes"] * 60:
                self.log(event="checkpoint", path=str(self.save()))
                last_save = time.time()
        self.log(event="checkpoint", path=str(self.save()), finished=self.state["tokens"] >= target)
