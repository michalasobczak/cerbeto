"""`cerbeto exam nursery`: BLiMP + conversation probe for a nursery checkpoint (spec 0003)."""
import json
import time
from pathlib import Path

import torch
import yaml

from cerbeto import paths
from cerbeto.exams import blimp, conversation
from cerbeto.model import checkpoint


def run(student: str, version: str | None, do_blimp: bool, do_conversation: bool, device: str) -> dict:
    ckpt_dir = paths.REPO / "checkpoints" / student
    version = version or (ckpt_dir / "LATEST").read_text().strip()
    model = checkpoint.load(ckpt_dir / version, device)
    k = yaml.safe_load((paths.CONFIG / "student.yaml").read_text())["recurrence"]["k_eval"]
    result = {"student": student, "version": version, "params": model.num_params(),
              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if do_blimp:
        result["blimp"] = blimp.evaluate(model, k, torch.device(device))
        print(f"BLiMP accuracy: {result['blimp']['accuracy']:.3f} (gate ≥ 0.65)")
    if do_conversation:
        result["conversation"] = conversation.evaluate(model, k, torch.device(device))
        print(f"Conversation probe: {result['conversation']['fitting']:.2f} fitting (gate ≥ 0.60)")
        for item in result["conversation"]["items"][::5]:
            print(f"  [{item['score']}] {item['say']!r} → {item['reply']!r}")
    out = paths.REPO / "runs" / "nursery" / student / f"exam_{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
