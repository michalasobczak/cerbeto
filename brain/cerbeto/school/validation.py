"""Validation set of a stage (spec 0005): picks the best moment of a night (early stopping).

A small, fixed set of tasks in the exam's forms — `id`, `ood_hard`, `ood_combo` — from its own seed range,
disjoint from the exam pools and excluded from lessons. The exam still decides promotion; validation only chooses
which weights of the night become the candidate, because on small models new combinations improve early in
training and then degrade while the lesson forms keep improving.
"""
from functools import lru_cache

from cerbeto.curriculum import EXAM_SEED_BASE, EXAM_VARIANTS, Task, generate, skills
from cerbeto.exams.grader import grade
from cerbeto.exams.school import answer, exam_hashes

SEED_BASE = EXAM_SEED_BASE + 5 * 10**8          # exams use EXAM_SEED_BASE + i; space probes + 10**8


@lru_cache(maxsize=8)
def validation_set(stage: str, per_cell: int = 20) -> tuple[Task, ...]:
    exam = exam_hashes()
    tasks, seen = [], set(exam)
    for s in skills(stage):
        for variant in EXAM_VARIANTS:
            got, seed = 0, SEED_BASE
            while got < per_cell and seed < SEED_BASE + 50 * per_cell:   # small task spaces give fewer
                t = generate(s.name, variant, seed)
                seed += 1
                if t.hash not in seen:
                    seen.add(t.hash)
                    tasks.append(t)
                    got += 1
    return tuple(tasks)


def validation_hashes(stages: list[str]) -> set[str]:
    return {t.hash for st in stages for t in validation_set(st)}


def score(model, tasks: tuple[Task, ...], k: int, device) -> float:
    replies = answer(model, [t.prompt for t in tasks], k, device)
    return sum(grade(t, r) for t, r in zip(tasks, replies)) / len(tasks)
