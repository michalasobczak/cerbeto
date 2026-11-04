"""A school day (spec 0005): lessons with the LLM teacher, graded by code, written to the hippocampus.

One lesson: the generator gives a task and its truth → for some lessons the homeroom teacher rephrases the question
in its own words (kept only if names, numbers and options survive and no negation appears) → the student answers
with the recurrent core → code grades → on a mistake the tutor solves and explains the task; its explanation is
used only when code verifies it, otherwise the generator's reference solution is the explanation → the experience
(prompt + the correct trace to learn) goes to the hippocampus. The teacher never sees an exam prompt (N1).
"""
import json
import random
import re
from collections.abc import Callable

from cerbeto import paths
from cerbeto.curriculum import Task, knowledge_words, training_stream, world
from cerbeto.curriculum.numbers import numbers_in
from cerbeto.exams.grader import grade
from cerbeto.exams.school import answer, exam_hashes
from cerbeto.memory.hippocampus import Experience, Hippocampus
from cerbeto.school.smoke import lesson_text
from cerbeto.school.validation import validation_hashes
from cerbeto.teachers import chat, teacher_config
from cerbeto.teachers.exam import SYSTEM as TUTOR, consistent, parse

Ask = Callable[[list[dict], bool], str]          # (messages, think) -> reply

REPHRASE = """Rewrite the question below in different simple words, the way a kind teacher talks to a young child.
Keep every name, every number, every option and the meaning exactly. Do not answer it. Reply with the new question
only."""
NO_REPHRASE = {"social.clarify"}   # a rephrasing may name the thing meant, and then "which one?" is no longer right
NEGATION = re.compile(r"\b(not|no|never)\b|n't", re.I)


def teacher(name: str | None = None) -> Ask:
    log = paths.REPO / "runs" / "teachers" / (name or "default") / "prompts.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    def ask(messages: list[dict], think: bool = False) -> str:
        with log.open("a") as f:                  # N1: everything the teacher was ever shown
            f.write(json.dumps({"prompt": messages[-1]["content"]}) + "\n")
        return chat(messages, name, temperature=0.3, think=think)
    return ask


def rephrase_ok(old: str, new: str) -> bool:
    """A rephrased question keeps names, numbers, options and knowledge-base words, and adds no negation."""
    if not new or len(new) > 3 * len(old) + 40 or "\n" in new.strip():
        return False
    if sorted(numbers_in(new.lower())) != sorted(numbers_in(old.lower())):
        return False
    w = world()
    known = set(w["names"]) | {n for names in w["pets"].values() for n in names}
    if not all(n in new for n in set(re.findall(r"\b[A-Z][a-z]+\b", old)) & known):
        return False
    if " or " in old and not all(o.strip(" ?.,").split()[-1] in new.lower()
                                  for o in old.lower().rsplit(",", 1)[-1].split(" or ") if o.strip(" ?.,")):
        return False
    low = new.lower()
    for word in set(re.findall(r"[a-z]+", old.lower())) & knowledge_words():
        if not re.search(rf"\b{word}(s|es)?\b", low):
            return False
    return not NEGATION.search(new) or bool(NEGATION.search(old))


def rephrased(task: Task, ask: Ask, exam: set[str]) -> Task:
    new = ask([{"role": "system", "content": REPHRASE}, {"role": "user", "content": task.question}], False).strip()
    new = new.strip('"').strip()
    if not rephrase_ok(task.question, new):
        return task
    t = Task.from_dict({**task.to_dict(), "question": new})
    return task if t.hash in exam else t


def tutor(task: Task, ask: Ask, think: bool) -> tuple[str, str] | None:
    """The teacher's verified (steps, answer) for a task, or None when code cannot confirm it."""
    reply = ask([{"role": "system", "content": TUTOR}, {"role": "user", "content": task.prompt}], think)
    steps, ans = parse(reply)
    if not steps or not grade(task, ans) or consistent(task, steps, ans) is False:
        return None
    return steps, ans


def target(task: Task, steps: str | None) -> str:
    """What the student learns: a thought (teacher's or reference steps) and the reference reply."""
    if steps and task.answer_type != "unknown":
        return f"<|think|> {steps} <|/think|><|answer|> {task.reply}<|eos|>"
    return lesson_text(task)[1]


def run_day(model, device, student: str, day: int, stage: str, cfg: dict, ask: Ask, teacher_name: str | None = None,
            stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print,
            hippo: Hippocampus | None = None) -> dict:
    hippo = hippo or Hippocampus()
    exam = exam_hashes() | validation_hashes([stage])      # lessons never show an exam or validation prompt
    think_skills = set(teacher_config(teacher_name).get("think_skills", []))
    rng = random.Random(f"day/{student}/{day}")
    lessons = training_stream(seed=10_000 + day, stage=stage, exclude=exam)
    done = hippo.count(student, day)              # resume a day interrupted half-way
    stats = {"lessons": done, "correct": 0, "rephrased": 0, "teacher_explained": 0, "code_explained": 0}
    while stats["lessons"] < cfg["lessons"] and not stop():
        given = [next(lessons) for _ in range(min(cfg["batch"], cfg["lessons"] - stats["lessons"]))]
        tasks = [rephrased(t, ask, exam) if t.skill not in NO_REPHRASE and t.answer_type != "unknown"
                 and rng.random() < cfg["rephrase_share"] else t for t in given]
        stats["rephrased"] += sum(t is not g for t, g in zip(tasks, given))
        replies = answer(model, [t.prompt for t in tasks], cfg["k"], device)
        rows = []
        for t, g, reply in zip(tasks, given, replies):
            ok = grade(t, reply)
            steps, origin, said = (" ".join(t.steps) or None), "code", {}
            if not ok and t.answer_type != "unknown" and not stop():
                explained = tutor(t, lambda m, th: said.setdefault("reply", ask(m, th)), t.skill in think_skills)
                if explained:
                    steps, origin = explained[0], "teacher"
            stats["correct"] += ok
            stats["teacher_explained" if origin == "teacher" else "code_explained"] += not ok
            prompt, _ = lesson_text(t)
            rows.append(Experience(student, day, stage, t.skill, origin, t.hash, prompt, target(t, steps), reply,
                                   ok, True, asked_as=None if t is g else g.question,
                                   teacher_reply=said.get("reply")))
        hippo.add(rows)
        stats["lessons"] += len(rows)
        log(f"day {day} · {stage} · {stats['lessons']}/{cfg['lessons']} lessons · correct "
            f"{stats['correct'] / max(1, stats['lessons'] - done):.0%} · teacher explained {stats['teacher_explained']}")
    return stats
