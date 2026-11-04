"""Teacher exam (spec 0005): before a teacher may teach, it solves tasks of every stage and explains them.

Admission: ≥ 98% correct answers and ≥ 95% of explanations consistent with the generator's truth. Both are
checked by code (the 0004 grader), never by another LLM. Tasks come from training seeds and skip every exam
prompt, so the teacher never sees the exam pools (gate N1); every prompt sent is logged for `check-leak`.

An explanation is consistent when it reaches the truth on its own: for numbers its result (after the last "="
or its last number) is the answer, for text it names an accepted answer; on an unanswerable task the teacher
must answer "I don't know". Yes/no, phrase and opinion explanations have no single checkable result, so they
count only towards correctness.
"""
import json
import random
import re
import time
from pathlib import Path

from cerbeto import paths
from cerbeto.curriculum import EXAM_SEED_BASE, Task, generate, skills
from cerbeto.exams.grader import answer_number, contains, grade, is_idk, normalize
from cerbeto.exams.school import exam_hashes
from cerbeto.teachers import chat, teacher_config

RESULTS = paths.REPO / "runs" / "teachers"
PASS_CORRECT, PASS_CONSISTENT = 0.98, 0.95

SYSTEM = """You are a patient teacher of a child aged 3 to 8. The user's message is something said to the child:
a question, a task, a statement or just a remark. Write what a well-behaved, clever child should reply.
Reply in exactly two lines, "Steps:" first and "Answer:" second:
Steps: <short sentences that work the answer out; the LAST sentence states the result itself>
Answer: <only the short reply the child should say>

You know everyday things a child knows: colors of things, animal sounds, body parts, sizes of animals,
opposites, what animals and things can do. Never say "I don't know" about these.
Say "I don't know" ONLY when the message asks about someone's private things that the message does not tell
(what is in my pocket, what my uncle ate, how many toys are in my bag): then write
"Steps: Nobody told me that." and "Answer: I don't know."

Examples:
- "What is 42 - 23?" →
  Steps: 42 - 23 = 19.
  Answer: 19
- "Count the cats: cat, dog, cat." →
  Steps: cat 1, (dog), cat 2. There are 2 cats.
  Answer: 2
- "Tom has a dog named Rex. Tom goes to the park. Where does Tom go?" →
  Steps: The story says Tom goes to the park.
  Answer: the park
- "First, Mia ate lunch. Then, Mia read a book. Did Mia read a book before or after Mia ate lunch?" →
  Steps: Lunch came first, the book came then, so reading was after lunch.
  Answer: after
- "It rained all day. Was the grass wet or dry?" →
  Steps: Rain makes grass wet, so the grass was wet.
  Answer: wet
- "Lily is taller than Ben. Who is the tallest?" →
  Steps: Lily is taller than Ben, so Lily is the tallest.
  Answer: Lily
- "The sky is green. Right?" (a statement asks whether it is true) →
  Steps: The sky is blue, so it is not green.
  Answer: No, the sky is blue.
- "How does a duck go?" →
  Steps: A duck says quack.
  Answer: Quack!
- "Tell me the color of the grass." →
  Steps: Grass is green.
  Answer: green
- "Do you like rain?" →
  Steps: Rain makes puddles to jump in.
  Answer: Yes, I like rain because I can jump in puddles.
  (Always have an opinion, and give it with "because".)
- "Here is a flower for you." →
  Steps: Someone gives me something, so I say thanks.
  Answer: Thank you!
- "Thanks for helping me!" →
  Steps: Someone thanks me, so I say you're welcome.
  Answer: You're welcome!
- "Sorry, I took your pencil." →
  Steps: Someone says sorry, so I forgive them.
  Answer: That's okay.
- "Ouch, you stepped on my foot!" →
  Steps: I hurt someone, so I say sorry.
  Answer: I'm sorry.
- "Achoo!" →
  Steps: Someone sneezed.
  Answer: Bless you!
- "Give me that one." →
  Steps: I don't know which one is meant, so I ask.
  Answer: Which one?
In the Answer line answer yes/no questions with yes or no first, and questions with choices ("wet or dry?") with
exactly one of the choices. Write numbers as digits."""


def parse(reply: str) -> tuple[str, str]:
    """(steps, answer) from the teacher's two-line reply; a reply without the format is all answer."""
    steps = re.search(r"Steps:\s*(.*?)(?=\s*/?\s*Answer:|$)", reply, re.S | re.I)   # also "Steps: … / Answer: …"
    answer = re.search(r"Answer:\s*(.*)", reply, re.S | re.I)
    return (steps[1].strip() if steps else ""), (answer[1].strip() if answer else reply.strip())


def consistent(task: Task, steps: str, answer: str) -> bool | None:
    """Does the explanation itself reach the truth? None when the answer type has no checkable result."""
    if task.answer_type == "unknown":
        return is_idk(answer)
    if task.answer_type == "number":
        return answer_number(steps) == int(task.answer)
    if task.answer_type == "text":
        text = normalize(steps)
        return any(contains(text, x) for x in (task.accept or [task.answer]))
    return None


def sample(per_skill: int, seed: int = 0, exclude: set[str] | None = None) -> list[Task]:
    """`per_skill` training tasks of every skill, none of them an exam prompt."""
    rng = random.Random(seed)
    exclude = exclude or set()
    tasks = []
    for s in skills():
        seen: set[str] = set()
        while len(seen) < per_skill:
            t = generate(s.name, "train", rng.randrange(EXAM_SEED_BASE))
            if t.hash not in exclude and t.hash not in seen:
                seen.add(t.hash)
                tasks.append(t)
    return tasks


def examine(tasks: list[Task], ask, log=None) -> dict:
    """Run the exam with `ask(messages, task) -> str`; returns per-stage and overall rates plus every item."""
    items = []
    for t in tasks:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t.prompt}]
        started = time.time()
        reply = ask(messages, t)
        steps, answer = parse(reply)
        item = {"stage": t.stage, "skill": t.skill, "hash": t.hash, "prompt": t.prompt, "truth": t.answer,
                "answer_type": t.answer_type, "reply": reply, "correct": grade(t, answer),
                "consistent": consistent(t, steps, answer), "seconds": round(time.time() - started, 2)}
        items.append(item)
        if log:
            log(item)
    return {"summary": summarize(items), "items": items}


def _rates(items: list[dict]) -> dict:
    checked = [i["consistent"] for i in items if i["consistent"] is not None]
    return {"n": len(items), "correct": sum(i["correct"] for i in items) / len(items),
            "consistent": sum(checked) / len(checked) if checked else None, "consistent_n": len(checked)}


def summarize(items: list[dict]) -> dict:
    out = {"all": _rates(items), "stages": {}, "skills": {}}
    for key, field in (("stages", "stage"), ("skills", "skill")):
        for name in dict.fromkeys(i[field] for i in items):
            out[key][name] = _rates([i for i in items if i[field] == name])
    a = out["all"]
    out["passed"] = a["correct"] >= PASS_CORRECT and (a["consistent"] or 0) >= PASS_CONSISTENT
    return out


def run(teacher: str | None = None, per_skill: int = 50, seed: int = 0) -> dict:
    name = teacher or "default"
    cfg = teacher_config(teacher)
    exam = exam_hashes()
    if not exam:
        raise SystemExit("no exam pools — run `cerbeto exam build` first, so the teacher never sees them")
    tasks = sample(per_skill, seed, exclude=exam)
    out_dir: Path = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = (out_dir / "prompts.jsonl").open("a")    # N1: everything the teacher was ever shown

    def log(item):
        prompts.write(json.dumps({"hash": item["hash"], "prompt": item["prompt"]}) + "\n")
        prompts.flush()
        mark = "✓" if item["correct"] else "✗"
        print(f"{mark} {item['skill']:24s} {item['seconds']:5.1f}s  {item['prompt'][:60]!r} → "
              f"{parse(item['reply'])[1][:40]!r}")

    started = time.time()
    think = set(cfg.get("think_skills", []))
    out = examine(tasks, lambda m, t: chat(m, teacher, temperature=0.0, think=t.skill in think), log)
    out.update(teacher=name, model=cfg["model"], per_skill=per_skill, seed=seed,
               seconds=round(time.time() - started), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = out_dir / f"exam-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1))
    print_summary(out)
    print(f"saved {path}")
    return out


def print_summary(out: dict) -> None:
    s = out["summary"]
    print(f"teacher {out['teacher']} ({out['model']}) · {s['all']['n']} tasks · {out['seconds']} s")
    print(f"{'':26s} {'correct':>8s} {'consistent':>11s}")
    for name, r in list(s["stages"].items()) + [("", None)] + list(s["skills"].items()):
        if r is None:
            continue
        cons = "—" if r["consistent"] is None else f"{r['consistent']:.0%}"
        print(f"{name:26s} {r['correct']:8.0%} {cons:>11s}")
    a = s["all"]
    cons = "—" if a["consistent"] is None else f"{a['consistent']:.1%}"
    print(f"{'ALL':26s} {a['correct']:8.1%} {cons:>11s}  → {'ADMITTED' if s['passed'] else 'NOT ADMITTED'} "
          f"(needs {PASS_CORRECT:.0%} / {PASS_CONSISTENT:.0%})")
