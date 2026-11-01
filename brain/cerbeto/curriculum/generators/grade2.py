"""Grade 2 — order of events, cause and effect, addition/subtraction up to 100, one-step word problems,
comparisons and relations (2–3 elements)."""
import random

from cerbeto.curriculum import skill, world
from cerbeto.curriculum.generators.common import numbers

STAGE = "grade2"
MIDDLE_WORDS = ["Then", "After that", "Next", "Later", "After a while"]


def _events(rng: random.Random, n: int) -> tuple[str, list[list[str]]]:
    name = rng.choice(world()["names"])
    evs = rng.sample(world()["events"], n)
    return name, evs


def _order_story(name: str, evs: list[list[str]]) -> str:
    words = ["First", "Then"] if len(evs) == 2 else ["First", *MIDDLE_WORDS[:len(evs) - 2], "In the end"]
    return " ".join(f"{w}, {name} {e[1]}." for w, e in zip(words, evs))


def _order_unknown(rng: random.Random, variant: str, difficulty: int) -> dict:
    name, evs = _events(rng, 4)
    missing, told = evs[0], evs[1:]
    return {"context": _order_story(name, told),
            "question": f"Did {name} {missing[0]} before or after {name} {told[0][1]}?",
            "steps": [f"The story does not say that {name} {missing[1]}."]}


@skill("order", STAGE, _order_unknown)
def order(rng: random.Random, variant: str, difficulty: int) -> dict:
    n = 7 if variant == "ood_hard" else {1: 2, 2: 3, 3: 3, 4: 4, 5: 5}[difficulty]
    name, evs = _events(rng, n)
    i, j = rng.sample(range(n), 2)
    context = _order_story(name, evs)
    if variant == "ood_combo":      # yes/no about the order
        rel = rng.choice(["before", "after"])
        truth = (i < j) == (rel == "before")
        return {"context": context, "question": f"Did {name} {evs[i][0]} {rel} {name} {evs[j][1]}?",
                "answer": "yes" if truth else "no", "answer_type": "yesno",
                "reply": f"{'Yes' if truth else 'No'}, {name} {evs[i][1]} {'before' if i < j else 'after'} that.",
                "steps": [f"{evs[i][1]} is number {i + 1}, {evs[j][1]} is number {j + 1}."]}
    ans = "before" if i < j else "after"
    return {"context": context, "question": f"Did {name} {evs[i][0]} before or after {name} {evs[j][1]}?",
            "answer": ans, "answer_type": "text", "accept": [ans], "reject": ["after" if ans == "before" else "before"],
            "reply": f"{name} {evs[i][1]} {ans} that.",
            "steps": [f"{evs[i][1]} is number {i + 1}, {evs[j][1]} is number {j + 1}.", f"So it was {ans}."]}


FORMS = {"melt": ["melted"], "freeze": ["froze", "frozen"], "break": ["broke", "broken"], "grow": ["grew", "grown"],
         "die": ["died", "dead"], "pop": ["popped"], "float": ["floated"]}


def _forms(word: str) -> list[str]:
    return [word, *FORMS.get(word, [])]


def _cause_unknown(rng: random.Random, variant: str, difficulty: int) -> dict:
    name = rng.choice(world()["names"])
    return {"context": f"{name} went to the park.",
            "question": rng.choice([f"Was {name} happy or sad?", f"Did {name} get wet or stay dry?",
                                    f"Was {name} hungry or full?"]),
            "steps": ["The story does not say how it went."]}


@skill("cause_effect", STAGE, _cause_unknown)
def cause_effect(rng: random.Random, variant: str, difficulty: int) -> dict:
    w = world()
    rare = set(w["causes_rare"])
    pool = sorted(rare) if variant == "ood_hard" else [i for i in range(len(w["causes"])) if i not in rare]
    cause, question, right, wrong, effect = w["causes"][rng.choice(pool)]
    name = rng.choice(w["names"])
    cause, question, effect = (x.replace("{n}", name) for x in (cause, question, effect))
    filler = ["It was a sunny day.", f"{name} has a little sister.", f"{name} likes apples."]
    context = " ".join(rng.sample(filler, min(2, max(0, difficulty - 2))) + [cause])
    if variant == "ood_combo":      # one option only: agree or disagree ("Did the glass break?")
        head = question[:-1].rsplit(" or ", 1)[0]
        stem = head.rsplit(" ", 1)[0]
        option = rng.choice([right, wrong])
        truth = option == right
        return {"context": cause, "question": f"{stem} {option}?", "answer": "yes" if truth else "no",
                "answer_type": "yesno", "reply": "Yes." if truth else f"No. {effect}",
                "steps": [f"{cause} So {effect[0].lower() + effect[1:]}"]}
    return {"context": context, "question": question, "answer": right, "answer_type": "text",
            "accept": _forms(right), "reject": _forms(wrong), "reply": effect, "steps": [f"{cause} So {effect[0].lower() + effect[1:]}"]}


SUM_MAX = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100, 6: 1000}


@skill("arithmetic100", STAGE, numbers)
def arithmetic100(rng: random.Random, variant: str, difficulty: int) -> dict:
    top = SUM_MAX[min(difficulty, 6)]
    lo = 101 if variant == "ood_hard" else 2
    total = rng.randint(lo, top)
    x = rng.randint(1, total - 1)
    if variant == "ood_combo":      # two operations
        y = rng.randint(1, x)
        q = f"What is {total - x} plus {x} minus {y}?"
        return {"question": q, "answer": str(total - y), "answer_type": "number",
                "reply": f"{total - x} + {x} - {y} = {total - y}.",
                "steps": [f"{total - x} + {x} = {total}", f"{total} - {y} = {total - y}"]}
    if rng.random() < 0.5:
        a_, op, b, ans = total - x, "+", x, total
    else:
        a_, op, b, ans = total, "-", x, total - x
    word = "plus" if op == "+" else "minus"
    q = rng.choice([f"What is {a_} {word} {b}?", f"What is {a_} {op} {b}?", f"{a_} {op} {b} = ?"])
    steps = []
    if op == "+" and a_ >= 10 and b >= 10:
        steps = [f"{a_} + {b // 10 * 10} = {a_ + b // 10 * 10}", f"{a_ + b // 10 * 10} + {b % 10} = {ans}"]
    elif op == "-" and b >= 10:
        steps = [f"{a_} - {b // 10 * 10} = {a_ - b // 10 * 10}", f"{a_ - b // 10 * 10} - {b % 10} = {ans}"]
    else:
        steps = [f"{a_} {op} {b} = {ans}"]
    return {"question": q, "answer": str(ans), "answer_type": "number", "reply": f"{a_} {op} {b} = {ans}.",
            "steps": steps}


WORD_MAX = {1: 10, 2: 20, 3: 40, 4: 70, 5: 100, 6: 1000}
THINGS = ["apples", "stickers", "marbles", "cookies", "flowers", "books", "balloons", "pencils"]


@skill("word_problems", STAGE, numbers)
def word_problems(rng: random.Random, variant: str, difficulty: int) -> dict:
    name, friend = rng.sample(world()["names"], 2)
    thing = rng.choice(THINGS)
    top = WORD_MAX[min(difficulty, 6)]
    total = rng.randint(101 if variant == "ood_hard" else 2, top)
    x = rng.randint(1, total - 1)
    if variant == "ood_combo":      # a relation between two children, then a question about the second one
        more = rng.random() < 0.5
        base = total - x if more else total
        other = base + x if more else base - x
        return {"context": f"{name} has {base} {thing}. {friend} has {x} {'more' if more else 'fewer'} {thing} than {name}.",
                "question": f"How many {thing} does {friend} have?", "answer": str(other), "answer_type": "number",
                "reply": f"{friend} has {other} {thing}.",
                "steps": [f"{base} {'+' if more else '-'} {x} = {other}"]}
    kind = rng.choice(["gets", "gives", "fly", "together"])
    if kind == "gets":
        ctx, q, ans, step = (f"{name} has {total - x} {thing}. {friend} gives {name} {x} more.",
                             f"How many {thing} does {name} have now?", total, f"{total - x} + {x} = {total}")
    elif kind == "gives":
        ctx, q, ans, step = (f"{name} has {total} {thing}. {name} gives {x} to {friend}.",
                             f"How many {thing} does {name} have now?", total - x, f"{total} - {x} = {total - x}")
    elif kind == "fly":
        ctx, q, ans, step = (f"There are {total} birds on a tree. {x} birds fly away.",
                             "How many birds are on the tree now?", total - x, f"{total} - {x} = {total - x}")
    else:
        ctx, q, ans, step = (f"{name} has {total - x} {thing}. {friend} has {x} {thing}.",
                             f"How many {thing} do they have together?", total, f"{total - x} + {x} = {total}")
    return {"context": ctx, "question": q, "answer": str(ans), "answer_type": "number",
            "reply": f"The answer is {ans}.", "steps": [step]}


ADJ = [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest")]


def _comparison_unknown(rng: random.Random, variant: str, difficulty: int) -> dict:
    a_, b, c, d = rng.sample(world()["names"], 4)
    comp, top, _ = rng.choice(ADJ)
    return {"context": f"{a_} is {comp} than {b}. {c} is {comp} than {d}.", "question": f"Who is the {top}?",
            "steps": [f"I know {a_} > {b} and {c} > {d}, but not how {a_} and {c} compare."]}


@skill("comparisons", STAGE, _comparison_unknown)
def comparisons(rng: random.Random, variant: str, difficulty: int) -> dict:
    n = 5 if variant == "ood_hard" else (2 if difficulty <= 2 else 3)
    people = rng.sample(world()["names"], n)          # people[0] is the most (tallest/oldest/fastest)
    comp, top, bottom = rng.choice(ADJ)
    if variant == "ood_combo":      # numbers + relation: ages
        base, older = rng.randint(4, 9), rng.randint(1, 3)
        a_, b, c = people[0], people[1], rng.choice([p for p in world()["names"] if p not in people[:2]])
        c_age = rng.choice([x for x in range(base - 2, base + older + 4) if x != base + older])
        ages = {b: base, a_: base + older, c: c_age}
        ans = max(ages, key=ages.get)
        return {"context": f"{b} is {base} years old. {a_} is {older} years older than {b}. {c} is {c_age} years old.",
                "question": "Who is the oldest?", "answer": ans, "answer_type": "text", "accept": [ans],
                "reject": [p for p in ages if p != ans], "reply": f"{ans} is the oldest.",
                "steps": [f"{a_} is {base} + {older} = {base + older}.", f"The oldest is {ans}."]}
    pairs = [(people[i], people[i + 1]) for i in range(n - 1)]
    rng.shuffle(pairs)
    sentences = [f"{x} is {comp} than {y}." for x, y in pairs]
    ask_top = rng.random() < 0.5
    ans = people[0] if ask_top else people[-1]
    word = top if ask_top else bottom
    return {"context": " ".join(sentences), "question": f"Who is the {word}?", "answer": ans, "answer_type": "text",
            "accept": [ans], "reject": [p for p in people if p != ans], "reply": f"{ans} is the {word}.",
            "steps": [" > ".join(people) + ".", f"The {word} is {ans}."]}
