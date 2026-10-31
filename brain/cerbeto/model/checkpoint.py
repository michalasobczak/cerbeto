"""Student checkpoints: one immutable directory per version.

    model.safetensors     fp32 weights (tied head is the embedding — stored once)
    architecture.json     StudentConfig + metadata (params, parent version, seed, git commit, time)
    optimizer.pt          AdamW state keyed by parameter name (optional)
    growth_log.jsonl      inherited growth/pruning events (copied from the parent, then appended)
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from cerbeto import paths
from cerbeto.model.growth import adamw_kwargs
from cerbeto.model.student import Student, StudentConfig


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save(model: Student, out: Path, optimizer: torch.optim.Optimizer | None = None, parent: Path | None = None,
         meta: dict | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=False)
    state = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    arch = {"config": model.cfg.to_dict(), "num_params": model.num_params(),
            "parent": str(parent) if parent else None, "git_commit": git_commit(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})}
    (out / "architecture.json").write_text(json.dumps(arch, indent=2))
    if parent and (parent / "growth_log.jsonl").exists():
        shutil.copy(parent / "growth_log.jsonl", out / "growth_log.jsonl")
    else:
        (out / "growth_log.jsonl").touch()
    if optimizer is not None:
        names = {id(p): n for n, p in model.named_parameters()}
        opt_state = {names[id(p)]: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()}
                     for p, s in optimizer.state.items()}
        groups = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
        torch.save({"state": opt_state, "groups": groups}, out / "optimizer.pt")
    return out


def load(path: Path, device: torch.device | str = "cpu", with_optimizer: bool = False):
    arch = json.loads((path / "architecture.json").read_text())
    model = Student(StudentConfig.from_dict(arch["config"]))
    model.load_state_dict(load_file(str(path / "model.safetensors")))
    model.to(device)
    if not with_optimizer:
        return model
    saved = torch.load(path / "optimizer.pt", map_location=device)
    opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs(saved["groups"][0]))
    params = dict(model.named_parameters())
    for name, st in saved["state"].items():
        opt.state[params[name]] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in st.items()}
    return model, opt
