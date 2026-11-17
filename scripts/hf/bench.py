"""Quick benchmark: 100 questions, each asked once, greedy (temperature 0, K = 1, CPU), an answer counts when it matches a pattern.
Faster and coarser than spot_check.py (16 questions × 10 samples); good for comparing several models side by side.
A model is a hub tag (v6-1032M) or a checkpoint folder relative to the repo; answers go to /tmp/bench/<model>.json.

    uv run --with huggingface_hub python ../scripts/hf/bench.py v6-1032M checkpoints/final/bil2-avg/final
"""
import json, re, shutil, sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from cerbeto import paths
from cerbeto.model import checkpoint
from cerbeto.talk import Talker

YES, NO = r"^(yes|true)", r"^(no|false)"
CHAT = {
    "yes/no": [("Is fire hot?", YES), ("Is ice cold?", YES), ("Is the sun hot?", YES), ("Is a dog an animal?", YES),
               ("Is a rose a plant?", YES), ("Can a fish swim?", YES), ("Can a bird fly?", YES), ("Can a cat fly?", NO),
               ("Is the sky green?", NO), ("Is snow hot?", NO), ("Do cows lay eggs?", NO), ("Is a whale a fish?", NO),
               ("A cat can fly. True or false?", NO), ("The sun is cold. True or false?", NO), ("Water is wet. True or false?", YES)],
    "preschool": [("What is the opposite of hot?", r"cold"), ("What is the opposite of big?", r"small|little"),
                  ("What is the opposite of up?", r"down"), ("What is the opposite of happy?", r"sad"),
                  ("What is the opposite of fast?", r"slow"), ("What is the opposite of day?", r"night"),
                  ("What is the opposite of open?", r"clos|shut"), ("What is the opposite of old?", r"new|young"),
                  ("What is the opposite of wet?", r"dry"), ("What is the opposite of full?", r"empty"),
                  ("What color is the sky?", r"blue"), ("What color is grass?", r"green"), ("What color is snow?", r"white"),
                  ("What color is a banana?", r"yellow"), ("What color is a tomato?", r"red"), ("What does a cow say?", r"moo"),
                  ("What does a dog say?", r"woof|bark|ruff"), ("What does a cat say?", r"meow|miaow|purr"),
                  ("What does a duck say?", r"quack"), ("Which is bigger, an elephant or a mouse?", r"elephant")],
    "knowledge": [("Where do fish live?", r"water|sea|river|lake|pond|ocean"), ("What do bees make?", r"honey"),
                  ("What do cows give us?", r"milk"), ("How many legs does a dog have?", r"four|\b4\b"),
                  ("How many legs does a bird have?", r"two|\b2\b"), ("How many legs does a spider have?", r"eight|\b8\b"),
                  ("What is the capital of France?", r"paris"), ("What is the capital of England?", r"london"),
                  ("What is the capital of Italy?", r"rome"), ("What is the capital of Japan?", r"tokyo"),
                  ("What is the largest planet?", r"jupiter"), ("What planet do we live on?", r"earth"),
                  ("What is the closest star to Earth?", r"\bsun\b"), ("What do plants need to grow?", r"water|sun|light"),
                  ("What gas do we breathe in?", r"oxygen"), ("What does the heart do?", r"pump|blood"),
                  ("What is ice made of?", r"water"), ("What do caterpillars turn into?", r"butterfl|moth"),
                  ("What is a baby dog called?", r"\bpup"), ("What is a baby cat called?", r"kitten"),
                  ("What is a baby cow called?", r"calf"), ("How many days are in a week?", r"seven|\b7\b"),
                  ("How many months are in a year?", r"twelve|\b12\b"), ("What season comes after winter?", r"spring"),
                  ("What do we use to see?", r"\beyes?\b"), ("What do we use to hear?", r"\bears?\b"),
                  ("Which animal is called the king of the jungle?", r"lion"),
                  ("What is the biggest animal in the ocean?", r"whale"),
                  ("Is a carrot a fruit or a vegetable?", r"vegetable"), ("Who wrote Romeo and Juliet?", r"shakespeare")],
    "arithmetic": [("What is one plus one?", r"two|\b2\b"), ("What is two plus three?", r"five|\b5\b"),
                   ("What is four plus four?", r"eight|\b8\b"), ("What is ten minus three?", r"seven|\b7\b"),
                   ("What is three plus five?", r"eight|\b8\b"), ("What is two plus two?", r"four|\b4\b"),
                   ("What is five minus two?", r"three|\b3\b"), ("What is six plus one?", r"seven|\b7\b"),
                   ("What is nine minus four?", r"five|\b5\b"), ("What is 7 plus 5?", r"twelve|\b12\b")]}
COMP = [("Water freezes and turns into", r"ice"), ("The Moon goes around the", r"earth"), ("The capital of France is", r"paris"),
        ("Bees make", r"honey"), ("The heart pumps", r"blood"), ("Plants need sunlight and", r"water"),
        ("A baby cat is called a", r"kitten"), ("The largest planet in our solar system is", r"jupiter"),
        ("The Earth goes around the", r"\bsun\b"), ("Fish breathe through their", r"gill"), ("Rain falls from the", r"cloud|sky"),
        ("A triangle has three", r"side|corner|angle|point"), ("The opposite of hot is", r"cold"), ("Cows give us", r"milk"),
        ("The color of the sky is", r"blue"), ("Birds have feathers and", r"wing|fly|beak|lay"), ("One, two, three, four,", r"five"),
        ("Monday, Tuesday,", r"wednesday"), ("January, February,", r"march"), ("Ice is frozen", r"water")]
MEM = [(["Hi! My name is Anna.", "I have a cat called Tom.", "What is my cat's name?"], r"tom"),
       (["My name is Jack.", "I have a horse called Sunny.", "What is my horse called?"], r"sunny"),
       (["I am Zoe.", "My bird is called Kiwi.", "What is my bird called?"], r"kiwi"),
       (["My name is Leo.", "I have a dog called Rex.", "What is my name?"], r"leo"),
       (["My name is Mia.", "My favourite color is red.", "What is my favourite color?"], r"red")]
assert sum(map(len, CHAT.values())) + len(COMP) + len(MEM) == 100


def fetch(v: str) -> Path:
    if re.fullmatch(r"v\d+-\d+M", v):
        return Path(snapshot_download("michalasobczak/cerbeto", revision=v, allow_patterns=["model.safetensors", "architecture.json"]))
    src, out = paths.REPO / v, Path("/tmp/bench/ckpt") / v.replace("/", "_")
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    for _ in range(5):                # `latest` is replaced every 30 minutes: weights and metadata from the same save
        a1 = (src / "architecture.json").read_text(); shutil.copy(src / "model.safetensors", out); a2 = (src / "architecture.json").read_text()
        if a1 == a2:
            break
    (out / "architecture.json").write_text(a2)
    return out


torch.set_num_threads(4)
names, res = sys.argv[1:], {}
for name in names:
    t = Talker(checkpoint.load(fetch(name), "cpu"), torch.device("cpu"), k=1, temperature=0, max_new_tokens=40)
    rows = []
    for part, qs in CHAT.items():
        for q, ok in qs:
            a = t.reply(q, remember=False)
            rows.append((part, q, a, bool(re.search(ok, a.lower()))))
    for c, ok in COMP:
        a = t.story("<|eos|>" + c, max_new_tokens=12).split(".")[0]
        rows.append(("complete", c + " …", a, bool(re.search(ok, a.lower()))))
    for conv, ok in MEM:
        t.history = ""
        for s in conv[:-1]:
            t.reply(s, remember=True)
        a = t.reply(conv[-1], remember=True)
        rows.append(("memory", " / ".join(conv), a, bool(re.search(ok, a.lower()))))
    res[name] = rows
    out = Path("/tmp/bench") / (name.replace("/", "_") + ".json"); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(name, "done", sum(r[3] for r in rows), "/ 100", flush=True)

print()
for i, (part, q, *_) in enumerate(res[names[0]]):
    print(f"{part:10} {q[:44]:44} " + " | ".join(f"{'✓' if res[n][i][3] else '✗'} {res[n][i][2][:38]:38}" for n in names))
print()
print(f"{'':11}" + "".join(f"{n[-22:]:>24}" for n in names))
for part in [*CHAT, "complete", "memory", "ALL"]:
    print(f"{part:11}" + "".join(f"{sum(r[3] for r in res[n] if part in ('ALL', r[0])):>20} / {sum(1 for r in res[n] if part in ('ALL', r[0])):<2}" for n in names))
