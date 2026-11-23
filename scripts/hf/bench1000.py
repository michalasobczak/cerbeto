"""The 1000-question benchmark: bench.py's 100 questions (core) and 900 more from bench1000_questions.py, asked the same way
(once, greedy, CPU; an answer counts when it matches a pattern; a question with two options counts when the right one comes
first). With 1000 questions a difference of ~3-4 points between two models is beyond chance; with 100 it was ~10.
A model is a hub tag of michalasobczak/cerbeto (v6-1032M, v15-12488M) or a checkpoint folder relative to the repo.
Answers go to /tmp/bench1000/<model>.json; the table at the end has the parts, the core 100 and all 1000.

    uv run --with huggingface_hub python ../scripts/hf/bench1000.py v6-1032M checkpoints/final/bil10/final
"""
import json, os, re, shutil, sys, time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from cerbeto import paths
from cerbeto.model import checkpoint
from cerbeto.talk import Talker

QS = json.loads(Path(__file__).with_name("bench1000.json").read_text())
PARTS = ["yes/no", "preschool", "knowledge", "arithmetic", "complete", "memory"]
OUT = Path("/tmp/bench1000")


def fetch(v: str) -> Path:
    if re.fullmatch(r"[a-z]+\d*-\d+M", v):
        return Path(snapshot_download("michalasobczak/cerbeto", revision=v, allow_patterns=["model.safetensors", "architecture.json"]))
    src, out = paths.REPO / v, OUT / "ckpt" / v.replace("/", "_")
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    for _ in range(5):                # `latest` is replaced every 30 minutes: weights and metadata from the same save
        a1 = (src / "architecture.json").read_text(); shutil.copy(src / "model.safetensors", out); a2 = (src / "architecture.json").read_text()
        if a1 == a2:
            break
    (out / "architecture.json").write_text(a2)
    return out


def right(item: dict, answer: str) -> bool:
    a = answer.lower()
    m = re.search(item["ok"], a)
    if not m or "not" not in item:
        return bool(m)
    w = re.search(item["not"], a)
    return not w or m.start() < w.start()


def ask(t: Talker, item: dict) -> str:
    if item["kind"] == "chat":
        return t.reply(item["q"], remember=False)
    if item["kind"] == "complete":
        return t.story("<|eos|>" + item["q"], max_new_tokens=12).split(".")[0]
    t.history = ""
    for s in item["q"][:-1]:
        t.reply(s, remember=True)
    return t.reply(item["q"][-1], remember=True)


torch.set_num_threads(4)
DEV = os.environ.get("BENCH_DEVICE", "cpu")    # cuda: a training card between rounds (minutes, not an hour of a busy CPU)
BENCH_K = int(os.environ.get("BENCH_K", "1"))  # passes of a looped core (spec 0021's students: 2)
OUT.mkdir(exist_ok=True)
names, res = sys.argv[1:], {}
for name in names:
    t0 = time.time()
    t = Talker(checkpoint.load(fetch(name), DEV), torch.device(DEV), k=BENCH_K, temperature=0, max_new_tokens=40)
    rows = []
    for item in QS:
        a = ask(t, item)
        rows.append({"part": item["part"], "core": item.get("core", False), "q": item["q"], "a": a, "ok": right(item, a)})
    res[name] = rows
    (OUT / (name.replace("/", "_") + ".json")).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"{name} done {sum(r['ok'] for r in rows)} / {len(rows)} in {time.time() - t0:.0f} s", flush=True)

print()
print(f"{'':11}" + "".join(f"{n[-22:]:>24}" for n in names))
for part in [*PARTS, "core 100", "ALL"]:
    pick = lambda r: part == "ALL" or (part == "core 100" and r["core"]) or r["part"] == part
    print(f"{part:11}" + "".join(f"{sum(r['ok'] for r in res[n] if pick(r)):>18} / {sum(1 for r in res[n] if pick(r)):<4}" for n in names))
