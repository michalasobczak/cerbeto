"""Final test (2026-11-06): does a larger student, or data in the request → reply form, make it usable in a talk?

Two arms start from the school's best version (v0017, 34.6M) and train on the same stream for the same tokens:
`control` keeps its size, `grown` is grown function-preservingly to ~99M (d 512 → 896, 28 heads, MLP 3072).
The stream alternates nursery shelf windows (language, all tokens learned) with request → reply pairs (only the
student's turn learned): a story told on request, preschool lessons (never an exam prompt) and who the student is.
Both arms and v0017 are then measured the same way: shelf validation loss, stories on request (held-out
phrasings, graded by the teacher), the nursery conversation probe and the preschool exam.

    python -m cerbeto.train.final_test train control --tokens 80e6
    python -m cerbeto.train.final_test eval control
"""
import argparse
import contextlib
import json
import math
import random
import re
import shutil
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from cerbeto import paths
from cerbeto.curriculum import training_stream
from cerbeto.exams import conversation
from cerbeto.exams import school as school_exam
from cerbeto.exams.school import exam_hashes
from cerbeto.model import checkpoint, growth
from cerbeto.model.student import pick_device
from cerbeto.school.smoke import IGNORE, encode_dialogues, lesson_text, release
from cerbeto.teachers import chat
from cerbeto.tokenizer import load as load_tokenizer
from cerbeto.train.data import ShelfStream

START = paths.REPO / "checkpoints" / "school" / "v0017"
CKPT = paths.REPO / "checkpoints" / "final"
RUN = paths.REPO / "runs" / "final"
SHELF_MIX = {"s0_stories": 0.40, "s0_speech": 0.30, "s1_books": 0.18, "s3_edu": 0.09, "s2_world": 0.03}
PAIR_MIX = {"story": 0.55, "lesson": 0.40, "self": 0.05}
CFG = {"seq_len": 512, "windows": 8, "pairs": 8, "pair_every": 3, "lr": 3e-4, "warmup": 300, "min_lr": 0.1,
       "weight_decay": 0.1, "k_min": 1, "k_max": 6, "k_eval": 4, "save_minutes": 30, "max_story_tokens": 400}
# passes of the core per arm: the deep arm has no loop (its core is unrolled), so K = 1 in training and evaluation
ARM_K = {"deep": {"k_min": 1, "k_max": 1, "k_eval": 1}, "mid": {"k_min": 1, "k_max": 1, "k_eval": 1}}
# request → reply mix per arm; `mid` adds conversations of several turns (the 283M student, trained on single pairs
# only, went on with a story when asked "who are you?" after it — 2026-11-06) and more about itself
ARM_PAIR_MIX = {"mid": {"story": 0.30, "lesson": 0.25, "self": 0.15, "dialogue": 0.30}}


def bf16_context(device, wanted: bool):
    """Mixed precision: forward and backward in bfloat16, weights and the optimizer stay fp32 (the checkpoint does
    not change). Only where the GPU computes bf16 natively (M6 yes, M1 no: it would emulate and be slower)."""
    if not wanted or str(device) == "cpu":
        return contextlib.nullcontext, False
    try:
        a = torch.ones(8, 8, device=device, dtype=torch.bfloat16)
        (a @ a).sum().item()
    except (RuntimeError, TypeError):
        return contextlib.nullcontext, False
    return (lambda: torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16)), True


def ks(arm: str) -> dict:
    return {k: CFG[k] for k in ("k_min", "k_max", "k_eval")} | ARM_K.get(arm, {})

# --- request → reply data ---------------------------------------------------------------------------
GENERIC = ["Tell me a story.", "Can you tell me a story?", "Tell me a story, please.", "I want a story.",
           "Please tell me a story.", "Will you tell me a story?", "tell me a story", "Story, please!"]
ABOUT_NAME = ["Tell me a story about {name}.", "Can you tell me a story about {name}?",
              "Please tell me a story about a {kind} named {name}.", "I want a story about {name}."]
ABOUT_KIND = ["Tell me a story about a {kind}.", "Can you tell me a story about a {kind}?",
              "Please tell me a story about a little {kind}.", "I want a story about a {kind}."]
SELF = [("Who are you?", "I'm Cerbeto. I'm a little student, and I'm learning to talk."),
        ("What are you?", "I'm a little student. My name is Cerbeto."),
        ("Are you a baby?", "No, I'm not a baby. I'm a little student."),
        ("Are you a person?", "No, I'm a little computer student. I'm learning to talk."),
        ("Are you an animal?", "No, I'm not an animal. I'm a little student."),
        ("What can you do?", "I can talk, answer easy questions and tell stories."),
        ("Can you tell stories?", "Yes! Ask me for a story."),
        ("Are you a robot?", "I'm a little computer student. I'm learning to talk."),
        ("What's your name?", "My name is Cerbeto."),
        ("What is your name?", "I'm Cerbeto."),
        ("How old are you?", "I'm very young. I'm still in preschool."),
        ("Do you go to school?", "Yes, I'm in preschool. I learn something new every day."),
        ("Do you like stories?", "Yes, I love stories!"),
        ("Who made you?", "People made me, and a teacher helps me learn."),
        ("Are you a cat?", "No, I'm not a cat. I'm a little student."),
        ("Are you a grown-up?", "No, I'm a little student. I'm still learning."),
        ("Can you talk?", "Yes, I can talk. I'm still learning new words."),
        ("Are you a girl or a boy?", "I'm not a girl or a boy. I'm a little computer student.")]
# questions about the story just told: the answer is in the context, not in the weights
RECALL = [("What was the {kind}'s name?", "The {kind}'s name was {name}."),
          ("What was the name of the {kind}?", "Its name was {name}."),
          ("Who was the story about?", "It was about a {kind} named {name}."),
          ("Who was in the story?", "A {kind} named {name}."),
          ("Was the story about a {kind}?", "Yes, it was about a {kind} named {name}.")]
# held out: conversations never seen in training; the last reply must contain one of the words
DIALOGUE_PROBE = [
    (["Tell me a story about a cat named Luna.", "Do you remember the cat's name?"], ["luna"]),
    (["Can you tell me a story about a boy named Sam?", "Who was that story about?"], ["sam"]),
    (["Tell me a story about a fox named Red.", "Was the story about a dog?"], ["no", "fox"]),
    (["Tell me a story.", "Who are you?"], ["cerbeto", "student"]),
    (["Tell me a story about a dog.", "What color is the sky?"], ["blue"]),
    (["Tell me a story about a bird named Pip.", "Are you a bird?"], ["no"]),
    (["Who are you?", "What color is grass?"], ["green"]),
    (["What color is the sky?", "Are you a baby?"], ["no"]),
    (["Is a cat an animal?", "What is your name?"], ["cerbeto"]),
    (["Tell me a story about a bear named Ben.", "Is a cat an animal?", "What was the bear's name?"], ["ben"]),
]
# held out: phrasings never seen in training (the probe for stories on request)
STORY_PROBE = ["Could you tell me a bedtime story?", "I'd love a story about a dog.",
               "Tell me a short story about a girl named Mia.", "Story time! What story do you have today?",
               "Can you make up a story about a little bird?", "Tell me a story about a bear who got lost.",
               "Do you know a story about a cat?", "Tell me a funny story.", "Tell me a story about a dragon.",
               "My little brother wants a story about a red ball. Can you tell one?"]
STORY_RUBRIC = """You grade a small student that is learning to talk. Someone asked it for a story.
Score 5 = a real story (characters, things happen, an ending) that fits what was asked; 4 = a story that mostly fits;
3 = a story-like text that ignores the request or breaks off; 2 = one or two sentences, or repeats itself;
1 = not a story at all. Answer only with JSON: {"score": <1-5>, "reason": "<short reason>"}"""


def stories(limit: int = 120_000, seed: int = 0) -> list[tuple[str, str, str | None, str | None, int]]:
    """(request, story, kind, name, tokens) from the stories shelf: a story told because someone asked for it;
    kind and name when the story has "a <kind> named <Name>" (for questions about it)."""
    tok, rng, out = load_tokenizer(), random.Random(seed), []
    with (paths.SHELVES / "s0_stories" / "train.jsonl").open() as f:
        for line in f:
            text = json.loads(line)["text"].strip()
            n = len(tok.encode(text).ids)
            if n > CFG["max_story_tokens"]:
                continue
            m = re.search(r"\b(?:a|an) (?:little |small |big |young |kind |old )?([a-z]+) named ([A-Z][a-z]+)", text)
            r = rng.random()
            if m and r < 0.45:
                ask = rng.choice(ABOUT_NAME).format(name=m[2], kind=m[1])
            elif m and r < 0.7:
                ask = rng.choice(ABOUT_KIND).format(kind=m[1])
            else:
                ask = rng.choice(GENERIC)
            out.append((f"<|human|> {ask}\n<|student|>", f"<|answer|> {text}<|eos|>",
                        m[1] if m else None, m[2] if m else None, n + 16))
            if len(out) >= limit:
                break
    return out


def turn(q: str, a: str) -> tuple[str, str]:
    return f"<|human|> {q}\n<|student|>", f"<|answer|> {a}<|eos|>"


class Pairs:
    """Batches of conversations for the pairs steps: most are one turn (a request and its reply), the `dialogue`
    share has 2–5 turns that change the subject — a story, then a question about it, about the student or a lesson —
    so the student learns to answer the last turn, using the earlier ones only when it is asked about them."""
    def __init__(self, seed: int = 0, mix: dict | None = None):
        self.rng = random.Random(seed)
        self.mix = mix or PAIR_MIX
        self.tok = load_tokenizer()
        self.stories = stories(seed=seed)
        self.named = [s for s in self.stories if s[3]]
        self.lessons = training_stream(seed=30_000 + seed, stage="preschool", exclude=exam_hashes(), lessons=True)

    def one(self, kind: str) -> tuple[str, str]:
        if kind == "story":
            return self.rng.choice(self.stories)[:2]
        if kind == "lesson":
            return lesson_text(next(self.lessons))
        return turn(*self.rng.choice(SELF))

    def dialogue(self, budget: int) -> list[tuple[str, str]]:
        turns, used = [], 0

        def add(t: tuple[str, str], n: int | None = None) -> bool:
            nonlocal used
            n = n if n is not None else len(self.tok.encode(t[0] + t[1]).ids)
            if used + n > budget:
                return False
            turns.append(t)
            used += n
            return True

        want = self.rng.randint(2, 5)
        if self.rng.random() < 0.6:                      # a story first, then questions about it or something else
            st = self.rng.choice(self.named)
            add(st[:2], st[4])
            if self.rng.random() < 0.7:
                q, a = self.rng.choice(RECALL)
                add(turn(q.format(kind=st[2]), a.format(kind=st[2], name=st[3])))
        while len(turns) < want:
            if not add(self.one(self.rng.choice(["lesson", "self", "lesson"]))):
                break
        if len(turns) < 2:                               # a story too long for anything after it
            add(self.one("self"))
        return turns

    def batch(self, n: int) -> list[list[tuple[str, str]]]:
        out = []
        for kind in self.rng.choices(list(self.mix), list(self.mix.values()), k=n):
            out.append(self.dialogue(CFG["seq_len"] - 8) if kind == "dialogue" else [self.one(kind)])
        return out


# --- arms -------------------------------------------------------------------------------------------
def dead_attention(model) -> list[int]:
    """Core blocks whose q and k are less than half the median of the others: their attention is even over every
    token (spread 1.00 in `cerbeto chat /stats`)."""
    norms = [float(b.attn.q.weight.norm() * b.attn.k.weight.norm()) for b in model.core]
    mid = sorted(norms)[len(norms) // 2]
    return [i for i, n in enumerate(norms) if n < 0.25 * mid]


def start_mid(model, probe, log):
    """~100M, deep and thin (like MobileLLM-125M): d 512 → 640, heads 16 → 20, the core run 4 times unrolled into
    16 blocks with 1% noise (copies that start apart), and every copy of a dead attention renewed:
    2 + 16 + 2 = 20 blocks at K = 1."""
    for op, kw in (("grow_model_dim", {"new_d": 640}), ("grow_width", {"extra_heads": 4})):
        model = growth.apply(model, op, probe, k=CFG["k_eval"], max_params=110_000_000, max_delta_loss=0.01,
                             log_path=log, reason="mid: ~100M deep", std=0.01, **kw)
    model = growth.apply(model, "unroll_core", probe, k=CFG["k_eval"], k_new=1, max_params=110_000_000,
                         max_delta_loss=0.02, log_path=log, reason="mid: unroll with noise", times=CFG["k_eval"],
                         noise=0.01)
    dead = dead_attention(model)
    for where in (dead, dead[1:]):                    # all copies; else keep one (its averaging) and renew the rest
        if not where:
            break
        try:
            return growth.apply(model, "renew_attention", probe, k=1, max_params=110_000_000, max_delta_loss=0.05,
                                log_path=log, reason="mid: dead attention (q, k ~3.5× small)", where=where)
        except growth.GrowthRejected:
            continue
    return model


def start_model(arm: str, device):
    model = checkpoint.load(START, device)
    if arm == "control":
        return model
    probe = ShelfStream("s0_speech", "valid", CFG["seq_len"]).batch(8, np.random.default_rng(1), device)
    log = RUN / arm / "growth_log.jsonl"
    if arm == "mid":
        return start_mid(model, probe, log)
    for op, kw in (("grow_model_dim", {"new_d": 896}), ("grow_width", {"extra_heads": 12, "extra_mlp": 1280})):
        model = growth.apply(model, op, probe, k=CFG["k_eval"], max_params=100_000_000, max_delta_loss=0.01,
                             log_path=log, reason="final test: ~100M", std=0.01, **kw)
    if arm == "deep":
        # like Qwen2.5-0.5B (24 blocks at d 896): the core run 4 times becomes 16 blocks (same function at K = 1),
        # then 4 new blocks with zero output: 2 + 20 + 2 = 24 blocks, ~283M parameters
        model = growth.apply(model, "unroll_core", probe, k=CFG["k_eval"], k_new=1, max_params=300_000_000,
                             max_delta_loss=0.01, log_path=log, reason="final test: deep like Qwen", times=CFG["k_eval"])
        for _ in range(4):
            model = growth.apply(model, "grow_depth", probe, k=1, max_params=300_000_000, max_delta_loss=0.01,
                                 log_path=log, reason="final test: deep like Qwen", std=0.01)
    return model


def lr_at(step: int, total: int) -> float:
    warm = min(1.0, (step + 1) / CFG["warmup"])
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))
    return CFG["lr"] * warm * (CFG["min_lr"] + (1 - CFG["min_lr"]) * cos)


def train(arm: str, tokens: float, seed: int = 0, bf16: bool = True) -> None:
    device = pick_device()
    run = RUN / arm
    run.mkdir(parents=True, exist_ok=True)
    resume = CKPT / arm / "latest"
    state = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else None
    if state and resume.exists():
        model, opt = checkpoint.load(resume, device, with_optimizer=True)
    else:
        model = start_model(arm, device)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], betas=(0.9, 0.95), weight_decay=CFG["weight_decay"])
        state = {"step": 0, "tokens": 0, "target_tokens": int(tokens), "params": model.num_params()}
        (run / "state.json").write_text(json.dumps(state))     # the panel knows the target before the first save
    # the 99M arm keeps only block inputs and recomputes the rest: ~20 GB of activations at k=6 would push a
    # 32 GB Mac into swap, which costs far more than the ~30% recomputation
    model.checkpointing = arm not in ("control", "mid")     # ~100M fits a 16 GB Mac without recomputation
    total_steps = int(state["target_tokens"] / (CFG["windows"] * CFG["seq_len"]))   # ~tokens per step, both kinds
    amp, bf16 = bf16_context(device, bf16)
    state["precision"] = "bf16" if bf16 else "fp32"
    tok = load_tokenizer()
    shelves = {s: ShelfStream(s, "train", CFG["seq_len"]) for s in SHELF_MIX}
    valid = {s: ShelfStream(s, "valid", CFG["seq_len"]).fixed_batches(8, 8, device, seed=1)
             for s in ("s0_speech", "s0_stories")}
    rng = np.random.default_rng(seed + state["step"])
    pairs = Pairs(seed + state["step"], ARM_PAIR_MIX.get(arm))
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    log = (run / "log.jsonl").open("a")
    print(f"{arm}: {model.num_params():,} parameters · step {state['step']}/{total_steps} · {device} · {state['precision']}", flush=True)
    model.train()
    saved, t0, t0_tokens, recent = time.time(), time.time(), state["tokens"], []
    kk = ks(arm)
    while state["step"] < total_steps and not stop["now"]:
        step = state["step"]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps)
        k = int(rng.integers(kk["k_min"], kk["k_max"] + 1))
        if step % CFG["pair_every"] == CFG["pair_every"] - 1:
            x, y = encode_dialogues(pairs.batch(CFG["pairs"]), tok, device, max_len=CFG["seq_len"])
            with amp():
                logits = model(x, k)
            loss = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=IGNORE)
            kind, n = "pairs", x.numel()
        else:
            name = rng.choice(list(SHELF_MIX), p=list(SHELF_MIX.values()))
            w = shelves[name].batch(CFG["windows"], rng, device)
            with amp():
                loss = model.loss(w[:, :-1], w[:, 1:], k)
            kind, n = "shelf", w[:, :-1].numel()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent.append((kind, loss.item()))
        state["step"] += 1
        state["tokens"] += n
        if state["step"] % 100 == 0:
            release(device)
            sh = [v for c, v in recent if c == "shelf"]
            pr = [v for c, v in recent if c == "pairs"]
            row = {"time": time.strftime("%H:%M:%S"), "step": state["step"], "tokens": state["tokens"],
                   "shelf_loss": round(sum(sh) / max(1, len(sh)), 4), "pair_loss": round(sum(pr) / max(1, len(pr)), 4),
                   "lr": round(lr_at(step, total_steps), 6), "tok_s": round((state["tokens"] - t0_tokens) / max(1, time.time() - t0))}
            recent = []
            if state["step"] % 1000 == 0:
                row["valid"] = validation(model, valid, kk["k_eval"])
                model.train()
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            state["tok_s_last"] = row["tok_s"]
        if time.time() - saved > CFG["save_minutes"] * 60:
            save(model, opt, arm, state)
            saved = time.time()
    save(model, opt, arm, state)
    if state["step"] >= total_steps:
        final = CKPT / arm / "final"
        if final.exists():
            shutil.rmtree(final)
        checkpoint.save(model, final, parent=START, meta={"final_test": {**state, "cfg": CFG, "shelves": SHELF_MIX,
                                                                          "pairs": ARM_PAIR_MIX.get(arm, PAIR_MIX)}})
        print(f"{arm}: finished · {final}", flush=True)
    else:
        print(f"{arm}: interrupted at step {state['step']} — run again to resume", flush=True)


def save(model, opt, arm: str, state: dict) -> None:
    tmp, latest = CKPT / arm / "latest.tmp", CKPT / arm / "latest"
    if tmp.exists():
        shutil.rmtree(tmp)
    checkpoint.save(model, tmp, optimizer=opt, parent=START, meta={"final_test": state})
    if latest.exists():
        shutil.rmtree(latest)
    tmp.rename(latest)
    (RUN / arm / "state.json").write_text(json.dumps(state))


@torch.no_grad()
def validation(model, valid: dict, k: int) -> dict:
    model.eval()
    out = {s: round(sum(model.loss(b[:, :-1], b[:, 1:], k).item() for b in bs) / len(bs), 4)
           for s, bs in valid.items()}
    release(next(model.parameters()).device)
    return out


# --- evaluation -------------------------------------------------------------------------------------
@torch.no_grad()
def tell(model, ask: str, device, k: int, max_new_tokens: int = 220) -> str:
    tok = load_tokenizer()
    ids = torch.tensor([[tok.token_to_id("<|eos|>")] + tok.encode(f"<|human|> {ask}\n<|student|>").ids], device=device)
    out = model.generate(ids, max_new_tokens, k=k, temperature=0.7, top_k=40)[0, ids.shape[1]:].tolist()
    stops = {tok.token_to_id(t) for t in ("<|eos|>", "<|human|>", "<|student|>")}
    cut = next((i for i, t in enumerate(out) if t in stops), len(out))
    return re.sub(r"<\|[^|]*\|>", "", tok.decode(out[:cut], skip_special_tokens=False)).strip()


def story_probe(model, device, k: int) -> dict:
    torch.manual_seed(0)
    items = []
    for ask in STORY_PROBE:
        reply = tell(model, ask, device, k)
        content = chat([{"role": "system", "content": STORY_RUBRIC},
                        {"role": "user", "content": f'Request: "{ask}"\nStudent: "{reply}"'}],
                       temperature=0.0, json_output=True)
        try:
            score = int(json.loads(content)["score"])
        except (ValueError, KeyError, TypeError):
            score = 1
        items.append({"ask": ask, "reply": reply, "score": score, "words": len(reply.split())})
    return {"mean_score": sum(i["score"] for i in items) / len(items),
            "stories": sum(i["score"] >= 4 for i in items) / len(items), "items": items}


def dialogue_probe(model, device, k: int) -> dict:
    """Held-out conversations, the history kept as in training; the last reply must contain an expected word."""
    from cerbeto.talk import Talker
    talker = Talker(model, device, k=k, temperature=0.0, max_new_tokens=80)
    talker.memory = True
    items = []
    for says, words in DIALOGUE_PROBE:
        talker.history = ""
        replies = [talker.reply(say) for say in says]
        ok = any(w in re.findall(r"[a-z]+", replies[-1].lower()) for w in words)
        items.append({"says": says, "replies": replies, "expected": words, "ok": ok})
    return {"accuracy": sum(i["ok"] for i in items) / len(items), "items": items}


def evaluate(name: str) -> dict:
    device = pick_device()
    path = START if name == "v0017" else CKPT / name / "final"
    if not path.exists():
        print(f"{name}: no finished training — skipped", flush=True)
        return {}
    model = checkpoint.load(path, device).eval()
    valid = {s: ShelfStream(s, "valid", CFG["seq_len"]).fixed_batches(8, 8, device, seed=1)
             for s in ("s0_speech", "s0_stories")}
    k = ks(name)["k_eval"]
    out = {"name": name, "params": model.num_params(), "k": k, "valid": validation(model, valid, k)}
    print(name, out["valid"], flush=True)
    out["dialogue"] = dialogue_probe(model, device, k)
    print(name, "dialogue", out["dialogue"]["accuracy"], flush=True)
    out["story"] = story_probe(model, device, k)
    print(name, "story", out["story"]["mean_score"], flush=True)
    torch.manual_seed(0)
    out["conversation"] = conversation.evaluate(model, k, device)
    print(name, "conversation", out["conversation"]["fitting"], flush=True)
    exam = school_exam.evaluate(model, k, device)
    from cerbeto.school.scheduler import stage_scores
    out["preschool"] = stage_scores(exam["results"]).get("preschool")
    out["exam"] = {s: {v: r[v]["accuracy"] for v in ("id", "ood_hard", "ood_combo")}
                   for s, r in exam["results"]["preschool"].items() if not s.startswith("_")}
    print(name, "preschool", out["preschool"], flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / f"eval_{name}.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["train", "eval"])
    p.add_argument("arm", help="control | grown | deep | mid (eval also: v0017)")
    p.add_argument("--tokens", type=float, default=80e6)
    p.add_argument("--fp32", action="store_true", help="no bf16 mixed precision")
    a = p.parse_args()
    train(a.arm, a.tokens, bf16=not a.fp32) if a.what == "train" else evaluate(a.arm)


if __name__ == "__main__":
    main()
