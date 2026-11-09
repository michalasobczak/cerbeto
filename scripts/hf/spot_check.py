"""Chat spot check of two versions: each question asked N times at temperature 0.7 (K = 1, CPU, so training is not
disturbed), an answer counts when it matches a pattern. A version is a hub tag (v3-523M) or a checkpoint folder;
`latest` is copied first, as it is replaced every 30 minutes. Before publishing a new version, compare it with the last one:

    uv run --with huggingface_hub python ../scripts/hf/spot_check.py v4-555M checkpoints/final/bil/latest
"""
import re, shutil, sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from cerbeto import paths
from cerbeto.model import checkpoint
from cerbeto.talk import Talker

N = 10
CHAT = [("Is fire hot?", r"^yes"), ("Is the sun hot?", r"^yes"), ("Is ice cold?", r"^yes"), ("Is snow cold?", r"^yes"),
        ("The sky is green. Is that right?", r"^no"), ("A cat can fly. True or false?", r"^(no|false)"),
        ("A fish can swim. True or false?", r"^(yes|true)"), ("Is a dog an animal?", r"^yes"),
        ("Is a carrot a fruit or a vegetable?", r"vegetable"), ("Where do fish live?", r"water|sea|river|lake|pond|ocean"),
        ("What do bees make?", r"honey"), ("How many legs does a bird have?", r"two|\b2\b"),
        ("What color is grass?", r"green"), ("What is the capital of France?", r"paris"),
        ("What do cows give us?", r"milk"), ("What is the opposite of hot?", r"cold")]
COMP = [("Water freezes and turns into", r"ice"), ("The Moon goes around the", r"earth"), ("The capital of France is", r"paris"),
        ("Bees make", r"honey"), ("The heart pumps", r"blood"), ("Plants need sunlight and", r"water"),
        ("A baby cat is called a", r"kitten"), ("The largest planet in our solar system is", r"jupiter")]
MEM = [(["Hi! My name is Anna.", "I have a cat called Tom.", "What is my cat's name?"], r"tom"),
       (["My name is Jack.", "I have a horse called Sunny.", "What is my horse called?"], r"sunny"),
       (["I am Zoe.", "My bird is called Kiwi.", "What is my bird called?"], r"kiwi"),
       (["My name is Leo.", "I have a dog called Rex.", "What is my name?"], r"leo")]



def fetch(v: str) -> Path:
    if re.fullmatch(r"v\d+-\d+M", v):
        return Path(snapshot_download("michalasobczak/cerbeto", revision=v, allow_patterns=["model.safetensors", "architecture.json"]))
    src, out = paths.REPO / v, Path("/tmp/spot_check") / Path(v).name   # relative to the repo, as in publish_model.py
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    for _ in range(5):                # weights and metadata from the same save
        a1 = (src / "architecture.json").read_text(); shutil.copy(src / "model.safetensors", out); a2 = (src / "architecture.json").read_text()
        if a1 == a2:
            break
    (out / "architecture.json").write_text(a2)
    return out


torch.set_num_threads(4)
A, B = sys.argv[1], sys.argv[2]
res = {}
for name, path in [(A, fetch(A)), (B, fetch(B))]:
    t = Talker(checkpoint.load(path, "cpu"), torch.device("cpu"), k=1, temperature=0.7, max_new_tokens=60)
    torch.manual_seed(0)
    r = {}
    for q, ok in CHAT:
        r[q] = sum(bool(re.search(ok, t.reply(q, remember=False).lower())) for _ in range(N))
    for c, ok in COMP:
        r["…" + c] = sum(bool(re.search(ok, t.story("<|eos|>" + c, max_new_tokens=12).lower().split(".")[0])) for _ in range(N))
    for conv, ok in MEM:
        n = 0
        for _ in range(N):
            t.history = ""
            for s in conv[:-1]:
                t.reply(s, remember=True)
            n += bool(re.search(ok, t.reply(conv[-1], remember=True).lower()))
        r["mem: " + conv[1]] = n
    res[name] = r
    print(name, "done", flush=True)
print(f"\n{'':52} A    B      A = {A}, B = {B}")
for k in res[A]:
    print(f"{k[:52]:52} {res[A][k]:>2}   {res[B][k]:>2}")
for part, pre in [("chat", None), ("complete", "…"), ("memory", "mem: ")]:
    keys = [k for k in res[A] if (k.startswith(pre) if pre else not (k.startswith("…") or k.startswith("mem: ")))]
    print(f"TOTAL {part:10} A {sum(res[A][k] for k in keys)}/{N*len(keys)}   B {sum(res[B][k] for k in keys)}/{N*len(keys)}")
