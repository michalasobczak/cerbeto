"""Grade 1 — reading comprehension, counting objects (up to 20), addition and subtraction within 10."""
import random

from cerbeto.curriculum import a, plural, skill, world
from cerbeto.curriculum.numbers import to_words
from cerbeto.curriculum.generators.common import numbers

STAGE = "grade1"


def _story(rng: random.Random, n_people: int, n_sentences: int) -> tuple[list[str], list[dict]]:
    """A template story about a few children: who has what, of which colour, which pet, where they go."""
    w = world()
    names = rng.sample(w["names"], n_people)
    objects = rng.sample(w["objects"], n_people)
    colors = rng.sample(w["color_words"][:8], n_people)
    places = rng.sample(w["places"], n_people)
    pets = rng.sample(sorted(w["pets"]), n_people)
    people = []
    for i, name in enumerate(names):
        people.append({"name": name, "object": objects[i], "color": colors[i], "place": places[i],
                       "pet": pets[i], "pet_name": rng.choice(w["pets"][pets[i]])})
    facts = []
    for p in people:
        facts += [(p, f"{p['name']} has {a(p['color'])} {p['object']}."),
                  (p, f"{p['name']} has {a(p['pet'])} named {p['pet_name']}."),
                  (p, f"{p['name']} goes to the {p['place']}.")]
    facts = facts[:max(n_sentences, n_people)]
    rng.shuffle(facts)
    return [s for _, s in facts], people


def _ask(rng: random.Random, sentences: list[str], people: list[dict]) -> dict:
    p = rng.choice([q for q in people if any(q["name"] in s for s in sentences)])
    known = [s for s in sentences if s.startswith(p["name"] + " ")]
    s = rng.choice(known)
    names = [q["name"] for q in people]
    if " named " in s:
        ans, q = p["pet_name"], f"What is the name of {p['name']}'s {p['pet']}?"
        reply = f"Its name is {ans}."
        reject = [x["pet_name"] for x in people if x is not p] + [p["name"]]
    elif " goes to " in s:
        ans, q = p["place"], f"Where does {p['name']} go?"
        reply = f"To the {ans}."
        reject = [x["place"] for x in people if x is not p]
    elif rng.random() < 0.5:
        ans, q = p["color"], f"What color is {p['name']}'s {p['object']}?"
        reply = f"It is {ans}."
        reject = [x["color"] for x in people if x is not p]
    else:
        ans, q = p["name"], f"Who has {a(p['color'])} {p['object']}?"
        reply = f"{ans} does."
        reject = [n for n in names if n != p["name"]]
    return {"question": q, "answer": ans, "answer_type": "text", "accept": [ans], "reject": reject,
            "reply": reply, "steps": [s]}


def _reading_unknown(rng: random.Random, variant: str, difficulty: int) -> dict:
    sentences, people = _story(rng, 1 if difficulty < 3 else 2, 2 + difficulty)
    p = people[0]
    q = rng.choice([f"What is the name of {p['name']}'s teacher?", f"What color is {p['name']}'s house?",
                    f"How old is {p['name']}?", f"What does {p['name']} eat for dinner?"])
    return {"context": " ".join(sentences), "question": q, "steps": ["The story does not say."]}


@skill("reading", STAGE, _reading_unknown)
def reading(rng: random.Random, variant: str, difficulty: int) -> dict:
    if variant == "ood_hard":
        sentences, people = _story(rng, 3, 9)
    else:
        sentences, people = _story(rng, 1 if difficulty < 3 else 2, 1 + difficulty)
    if variant == "ood_combo":      # counting inside a story
        p = people[0]
        count = rng.randint(2, 5)
        sentences = sentences + [f"{p['name']} has {to_words(count)} {plural('apple')}."]
        q = f"How many apples does {p['name']} have?"
        return {"context": " ".join(sentences), "question": q, "answer": str(count), "answer_type": "number",
                "reply": f"{p['name']} has {count} apples.", "steps": [sentences[-1]]}
    return {"context": " ".join(sentences), **_ask(rng, sentences, people)}


COUNT_MAX = {1: 5, 2: 8, 3: 10, 4: 15, 5: 20, 6: 30}


@skill("counting", STAGE, numbers)
def counting(rng: random.Random, variant: str, difficulty: int) -> dict:
    w = world()
    animals = ["cat", "dog", "duck", "cow", "bird", "pig", "frog", "sheep"]
    target = rng.choice(animals)
    hi = COUNT_MAX[min(difficulty, 6)]
    lo = 21 if variant == "ood_hard" else 1
    n = rng.randint(lo, hi)
    others = rng.sample([x for x in animals if x != target], 2 if difficulty >= 3 else (1 if difficulty == 2 else 0))
    items = [target] * n + [rng.choice(others) for _ in range(rng.randint(0, n) if others else 0)]
    rng.shuffle(items)
    if variant == "ood_combo":      # counting inside a story, across two groups
        name = rng.choice(w["names"])
        first = rng.randint(1, 5)
        n = first + rng.randint(1, 5)
        context = (f"{name} goes to the farm. {name} sees {to_words(first)} {plural(target) if first > 1 else target} "
                   f"in the field and {to_words(n - first)} {plural(target) if n - first > 1 else target} near the barn.")
        return {"context": context, "question": f"How many {plural(target)} does {name} see?", "answer": str(n),
                "answer_type": "number", "reply": f"{name} sees {n} {plural(target)}.",
                "steps": [f"{first} + {n - first} = {n}"]}
    q = rng.choice([f"Count the {plural(target)}: {', '.join(items)}. How many {plural(target)} are there?",
                    f"Here are some animals: {', '.join(items)}. How many {plural(target)} do you see?"])
    seen = 0
    steps = []
    for x in items:
        if x == target:
            seen += 1
            steps.append(f"{x} {seen}")
    return {"question": q, "answer": str(n), "answer_type": "number",
            "reply": f"There {'is' if n == 1 else 'are'} {n} {target if n == 1 else plural(target)}.",
            "steps": [", ".join(steps)]}


ADD_MAX = {1: 3, 2: 5, 3: 7, 4: 9, 5: 10, 6: 20}


@skill("arithmetic10", STAGE, numbers)
def arithmetic10(rng: random.Random, variant: str, difficulty: int) -> dict:
    top = ADD_MAX[min(difficulty, 6)]
    while True:
        total = rng.randint(1, top)
        x = rng.randint(0, total)
        if variant != "ood_hard" or total > 10:
            break
    plus = rng.random() < 0.5
    if plus:
        a_, b, ans, op, word = x, total - x, total, "+", "plus"
    else:
        a_, b, ans, op, word = total, x, total - x, "-", "minus"
    if variant == "ood_combo":      # numbers and operation in words
        word = rng.choice(["plus", "and"]) if plus else rng.choice(["minus", "take away"])
        q = f"What is {to_words(a_)} {word} {to_words(b)}?"
    else:
        q = rng.choice([f"What is {a_} {word} {b}?", f"What is {a_} {op} {b}?", f"{a_} {op} {b} = ?"])
    return {"question": q, "answer": str(ans), "answer_type": "number", "reply": f"{a_} {op} {b} = {ans}.",
            "steps": [f"{a_} {op} {b} = {ans}"]}
