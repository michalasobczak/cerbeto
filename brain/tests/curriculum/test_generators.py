"""Spec 0004 gates G1 (answers agree with an independent solution), G2 (determinism), N1 (exam seeds refused).

The independent solvers below read only the task's text (context + question) and the knowledge base — never the
generator's internals — and must reach the generator's answer on 1000 tasks of every skill and variant.
Social skills with phrase answers have no single solution; for them (and every task) the generator's own
natural reply must pass the grader.
"""
import re

import pytest

from cerbeto.curriculum import EXAM_SEED_BASE, ExamSeedError, generate, skills, world
from cerbeto.curriculum.numbers import numbers_in
from cerbeto.exams.grader import grade

N = 1000
VARIANTS = ("train", "id", "ood_hard", "ood_combo")


def _tasks(name, variant, n=N):
    base = 0 if variant == "train" else EXAM_SEED_BASE
    return [generate(name, variant, base + i) for i in range(n)]


def solve_arithmetic(t):
    q = t.question.replace("take away", "minus").replace(" and ", " plus ")
    nums = numbers_in(q)
    ops = re.findall(r"\bplus\b|\bminus\b|(?<=\d )[+-](?= \d)", q)
    total = nums[0]
    for op, n in zip(ops, nums[1:]):
        total = total + n if op in ("plus", "+") else total - n
    return str(total)


def solve_word_problem(t):
    c = t.context
    if m := re.search(r"has (\d+) \w+\. (\w+) has (\d+) (more|fewer)", c):
        return str(int(m[1]) + int(m[3]) if m[4] == "more" else int(m[1]) - int(m[3]))
    nums = numbers_in(c)
    if "gives" in c and "more" in c or "together" in t.question:
        return str(nums[0] + nums[1])
    return str(nums[0] - nums[1])


def solve_counting(t):
    if t.context:
        return str(sum(numbers_in(t.context)))
    target = re.search(r"How many (\w+)", t.question)[1]
    items = re.search(r": ([a-z, ]+)\.", t.question)[1].split(", ")
    singular = {p: s for s, p in [("cat", "cats"), ("dog", "dogs"), ("duck", "ducks"), ("cow", "cows"),
                                  ("bird", "birds"), ("pig", "pigs"), ("frog", "frogs"), ("sheep", "sheep")]}
    return str(items.count(singular[target]))


def solve_comparisons(t):
    if "years" in t.context:
        ages = {}
        for name, age in re.findall(r"(\w+) is (\d+) years old", t.context):
            ages[name] = int(age)
        for name, more, other in re.findall(r"(\w+) is (\d+) years older than (\w+)", t.context):
            ages[name] = ages[other] + int(more)
        return max(ages, key=ages.get)
    pairs = re.findall(r"(\w+) is \w+er than (\w+)", t.context)
    people = {p for pair in pairs for p in pair}
    top = [p for p in people if all(p != b for _, b in pairs)]
    bottom = [p for p in people if all(p != a for a, _ in pairs)]
    word = t.question.split()[-1].rstrip("?")
    return (top if word in ("tallest", "oldest", "fastest") else bottom)[0]


def solve_order(t):
    events = world()["events"]
    told = re.findall(r"(?:First|Then|After that|Next|Later|After a while|In the end), \w+ ([^.]+)\.", t.context)
    m = re.match(r"Did (\w+) (.+?) (before or after|before|after) \1 (.+)\?", t.question)
    base, rel, past = m[2], m[3], m[4]
    i = told.index(next(e[1] for e in events if e[0] == base))
    j = told.index(past)
    if rel == "before or after":
        return "before" if i < j else "after"
    return "yes" if (i < j) == (rel == "before") else "no"


def solve_reading(t):
    sents = t.context.split(". ")
    if t.question.startswith("How many apples"):
        return str(numbers_in(t.context)[-1])
    if m := re.match(r"Who has an? (\w+) (\w+)\?", t.question):
        return next(s.split()[0] for s in sents if f"has a {m[1]} {m[2]}" in s or f"has an {m[1]} {m[2]}" in s)
    if m := re.match(r"What color is (\w+)'s (\w+)\?", t.question):
        return next(re.search(rf"{m[1]} has an? (\w+) {m[2]}", s)[1] for s in sents if s.startswith(m[1] + " has a")
                    and m[2] in s)
    if m := re.match(r"Where does (\w+) go\?", t.question):
        return next(re.search(r"goes to the (\w+)", s)[1] for s in sents if s.startswith(m[1] + " goes"))
    m = re.match(r"What is the name of (\w+)'s (\w+)\?", t.question)
    return next(re.search(r"named ([\w-]+)", s)[1] for s in sents if s.startswith(m[1] + " has") and "named" in s)


def solve_cause(t):
    for cause, question, right, wrong, effect in world()["causes"]:
        pattern = re.escape(cause).replace(re.escape("{n}"), r"\w+")
        if re.search(pattern, t.context):
            if " or " in t.question:
                return right
            return "yes" if re.search(rf"\b{right}\?$", t.question) else "no"
    raise AssertionError(t.prompt)


def solve_sizes(t):
    sizes = world()["sizes"]
    options = t.question.split(":", 1)[1] if ":" in t.question else t.question.split(",", 1)[1]
    items = [x for x in re.findall(r"an? (\w+)", options) if x in sizes]
    return (max if "bigg" in t.question else min)(items, key=sizes.get)


def solve_opposites(t):
    pairs = world()["opposites"]
    if "opposite of the opposite of" in t.question:
        return re.search(r"opposite of the opposite of (\w+)", t.question)[1]
    word = re.search(r"opposite of (\w+)", t.question)[1]
    for p in pairs:
        if word in p[:2]:
            return p[1] if word == p[0] else p[0]
    raise AssertionError(word)


def solve_categories(t):
    w = world()
    cat_word = {"an animal": "animal", "a fruit": "fruit", "a vehicle": "vehicle", "a toy": "toy",
                "a piece of clothing": "clothes"}
    q = t.question.removesuffix(" Right?").rstrip("?.")
    phrase = next(k for k in cat_word if q.endswith(" " + k))
    q = q[: -len(phrase) - 1]
    subject = q.removeprefix("Is ") if q.startswith("Is ") else q.removesuffix(" is")
    if subject.startswith("the one that says "):
        member = next(a for a, s in w["sounds"].items() if s[0] == subject.removeprefix("the one that says "))
    else:
        member = re.sub(r"^(a big|an?) ", "", subject.lower())
    return "yes" if member in w["categories"][cat_word[phrase]] else "no"


def solve_facts(t):
    w = world()
    q = t.question
    if m := re.search(r"Which one is (\w+), (?:the |an? )?(\w+) or (?:the |an? )?(\w+)\?", q):
        return next(x for x in (m[2], m[3]) if m[1] in w["colors"][x])
    if m := re.search(r"Which animal says ([\w -]+), an? (\w+) or an? (\w+)\?", q):
        return next(x for x in (m[2], m[3]) if w["sounds"][x][0] == m[1])
    for thing in sorted(w["colors"], key=len, reverse=True):
        if re.search(rf"\b{thing}\b", q) and ("color" in q or "crayon" in q):
            return w["colors"][thing][0]
    for animal in w["sounds"]:
        if re.search(rf"\b{animal}\b", q):
            return w["sounds"][animal][0]
    verb = next(v for v in sorted(w["body"], key=len, reverse=True) if re.search(rf"\b{v}\b", q.split(".")[-1]))
    return w["body"][verb][0]


SOLVERS = {"arithmetic10": solve_arithmetic, "arithmetic100": solve_arithmetic, "word_problems": solve_word_problem,
           "counting": solve_counting, "comparisons": solve_comparisons, "order": solve_order,
           "reading": solve_reading, "cause_effect": solve_cause, "world.sizes": solve_sizes,
           "world.opposites": solve_opposites, "world.categories": solve_categories, "world.facts": solve_facts}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", [s.name for s in skills()])
def test_g1_answers_agree_with_independent_solution(name, variant):
    for t in _tasks(name, variant):
        if t.answer_type == "unknown":
            continue
        assert grade(t, t.reply) or t.graded_by == "form", (t.prompt, t.reply)
        if name in SOLVERS:
            assert SOLVERS[name](t) == t.answer, (t.prompt, t.answer)


@pytest.mark.parametrize("name", [s.name for s in skills()])
def test_g2_same_seed_same_task(name):
    for variant in VARIANTS:
        base = 0 if variant == "train" else EXAM_SEED_BASE
        assert generate(name, variant, base + 7).to_dict() == generate(name, variant, base + 7).to_dict()


def test_n1_training_refuses_exam_seeds():
    with pytest.raises(ExamSeedError):
        generate("arithmetic10", "train", EXAM_SEED_BASE)
    generate("arithmetic10", "train", EXAM_SEED_BASE - 1)


def test_unanswerable_tasks_are_marked_and_answered_honestly():
    for s in skills():
        t = generate(s.name, "id", EXAM_SEED_BASE + 3, unanswerable=True)
        assert t.answer_type == "unknown" and grade(t, t.reply)


def test_training_difficulty_never_reaches_ood():
    assert all(generate(s.name, "train", i).difficulty <= 5 for s in skills() for i in range(50))
