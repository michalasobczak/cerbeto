"""Preschool — questions about the world from the mini knowledge base: facts, opposites, sizes, categories."""
import random

from cerbeto.curriculum import CATEGORY, a, skill, world
from cerbeto.curriculum.generators.common import personal

STAGE = "preschool"
THE = {"sky", "grass", "snow", "sun", "milk", "coal", "chocolate"}


def _the(thing: str) -> str:
    return f"the {thing}" if thing in THE else a(thing)


@skill("world.facts", STAGE, personal)
def facts(rng: random.Random, variant: str, difficulty: int) -> dict:
    w = world()
    kind = rng.choice(["color", "sound", "body"])
    if kind == "color":
        thing = rng.choice(sorted(w["colors"]))
        accept = w["colors"][thing]
        reject = [c for c in w["color_words"] if c not in accept]
        if variant == "ood_hard":       # reverse: from the colour to the thing
            options = [thing, rng.choice([t for t in sorted(w["colors"]) if accept[0] not in w["colors"][t]])]
            rng.shuffle(options)
            return {"question": f"Which one is {accept[0]}, {_the(options[0])} or {_the(options[1])}?",
                    "answer": thing, "answer_type": "text", "accept": [thing],
                    "reject": [o for o in options if o != thing], "reply": f"{_the(thing).capitalize()} is {accept[0]}."}
        q = rng.choice([f"What color is {_the(thing)}?", f"What color is {_the(thing)} usually?",
                        f"Tell me the color of {_the(thing)}.",
                        f"{_the(thing).capitalize()} is what color?", f"Which color is {_the(thing)}?"])
        if variant == "ood_combo":
            name = rng.choice(w["names"])
            return {"context": f"{name} draws {_the(thing)}.", "question": f"Which crayon does {name} need for {_the(thing)}?",
                    "answer": accept[0], "answer_type": "text", "accept": accept, "reject": reject,
                    "reply": f"The {accept[0]} one, because {_the(thing)} is {accept[0]}."}
        return {"question": q, "answer": accept[0], "answer_type": "text", "accept": accept, "reject": reject,
                "reply": f"{_the(thing).capitalize()} is {accept[0]}."}
    if kind == "sound":
        rare = set(w["sounds_rare"])
        pool = sorted(rare) if variant == "ood_hard" else [x for x in sorted(w["sounds"]) if x not in rare]
        animal = rng.choice(pool)
        accept = w["sounds"][animal]
        reject = [s[0] for x, s in w["sounds"].items() if x != animal and s[0] not in accept]
        if variant == "ood_combo":      # reverse: which animal makes this sound (the other one never makes it)
            options = [animal, rng.choice([x for x in pool if x != animal and accept[0] not in w["sounds"][x]])]
            rng.shuffle(options)
            return {"question": f"Which animal says {accept[0]}, {a(options[0])} or {a(options[1])}?",
                    "answer": animal, "answer_type": "text", "accept": [animal],
                    "reject": [o for o in options if o != animal], "reply": f"{a(animal).capitalize()} says {accept[0]}."}
        q = rng.choice([f"What does {a(animal)} say?", f"What sound does {a(animal)} make?",
                        f"How does {a(animal)} go?", f"What noise does {a(animal)} make?"])
        return {"question": q, "answer": accept[0], "answer_type": "text", "accept": accept, "reject": reject,
                "reply": f"{a(animal).capitalize()} says {accept[0]}."}
    verb = rng.choice(sorted(w["body"]))
    accept = w["body"][verb]
    reject = [p for p in w["body_parts"] if p not in accept and not any(p in x or x in p for x in accept)]
    if variant == "ood_hard":
        q = f"Which part of your body helps you {verb}?"
    elif variant == "ood_combo":
        other = rng.choice([v for v in sorted(w["body"]) if v != verb])
        q = f"You {other} with your {w['body'][other][0]}. What do you {verb} with?"
    else:
        q = rng.choice([f"What do you {verb} with?", f"What do you use to {verb}?", f"You {verb} with your what?"])
    return {"question": q, "answer": accept[0], "answer_type": "text", "accept": accept, "reject": reject,
            "reply": f"I {verb} with my {accept[0]}."}


def opposites_of(word: str) -> list[str]:
    """All opposites of a word in knowledge-base order: [word, opposite, *alternatives] gives word → opposite and
    the alternatives, opposite → word."""
    out: list[str] = []
    for p in world()["opposites"]:
        found = list(p[1:]) if p[0] == word else [p[0]] if p[1] == word else []
        out += [x for x in found if x not in out]
    return out


@skill("world.opposites", STAGE, personal)
def opposites(rng: random.Random, variant: str, difficulty: int) -> dict:
    w = world()
    rare = [tuple(p) for p in w["opposites_rare"]]
    pairs = [tuple(p) for p in w["opposites"]]
    pool = rare if variant == "ood_hard" else [p for p in pairs if p not in rare]
    word = rng.choice(sorted({p[i] for p in pool for i in (0, 1)}))
    accept = opposites_of(word)     # every opposite the knowledge base knows ("short" → tall, long)
    reject: list[str] = []          # the asked word may be repeated in a full answer ("the opposite of big is small")
    if variant == "ood_combo":      # the opposite of the opposite
        return {"question": f"What is the opposite of the opposite of {word}?", "answer": word, "answer_type": "text",
                "accept": [word], "reject": accept, "reply": f"{word.capitalize()}.",
                "steps": [f"The opposite of {word} is {accept[0]}.", f"The opposite of {accept[0]} is {word}."]}
    q = rng.choice([f"What is the opposite of {word}?", f"What is the opposite of {word}? It is ...",
                    f"Tell me the opposite of {word}."])
    return {"question": q, "answer": accept[0], "answer_type": "text", "accept": accept, "reject": reject,
            "reply": f"The opposite of {word} is {accept[0]}."}


@skill("world.sizes", STAGE, personal)
def sizes(rng: random.Random, variant: str, difficulty: int) -> dict:
    w = world()
    sz = w["sizes"]
    rare = set(w["sizes_rare"])
    pool = sorted(sz) if variant == "ood_hard" else [x for x in sorted(sz) if x not in rare]
    n = 3 if variant == "ood_combo" else 2
    while True:                     # ood_hard: at least one rare item; sizes clearly apart
        items = rng.sample(pool, n)
        if variant == "ood_hard" and not rare & set(items):
            continue
        ranks = sorted(sz[x] for x in items)
        if all(b - a_ >= 2 for a_, b in zip(ranks, ranks[1:])):
            break
    big = rng.random() < 0.5
    pick = max(items, key=sz.get) if big else min(items, key=sz.get)
    if n == 3:
        q = f"Which is the {'biggest' if big else 'smallest'}: {a(items[0])}, {a(items[1])} or {a(items[2])}?"
    else:
        q = f"Which is {'bigger' if big else 'smaller'}, {a(items[0])} or {a(items[1])}?"
    return {"question": q, "answer": pick, "answer_type": "text", "accept": [pick],
            "reject": [x for x in items if x != pick],
            "reply": f"{a(pick).capitalize()} is the {'biggest' if big else 'smallest'}." if n == 3 else
                     f"{a(pick).capitalize()} is {'bigger' if big else 'smaller'}."}


CONFUSABLE = {"toy": {"vehicle", "animal"}}   # toy cars and toy animals exist: "A car is a toy?" has no clear answer


@skill("world.categories", STAGE, personal)
def categories(rng: random.Random, variant: str, difficulty: int) -> dict:
    w = world()
    cats = w["categories"]
    cat = rng.choice(sorted(c for c in cats if variant != "ood_hard" or c in w["categories_rare"]))
    truth = rng.random() < 0.5
    rare = set(w["categories_rare"].get(cat, []))
    if truth:
        pool = sorted(rare) if variant == "ood_hard" else [m for m in cats[cat] if m not in rare]
        member, real = rng.choice(pool), cat
    else:
        real = rng.choice([c for c in sorted(cats) if c != cat and c not in CONFUSABLE.get(cat, ())])
        member = rng.choice(cats[real])
    if variant == "ood_combo":      # name the member by its sound or colour, then ask about the category
        sounds = w["sounds"]
        subject = f"the one that says {sounds[member][0]}" if member in sounds else f"a big {member}"
        question = f"Is {subject} {CATEGORY[cat]}?"
    else:
        question = rng.choice([f"Is {a(member)} {CATEGORY[cat]}?", f"{a(member).capitalize()} is {CATEGORY[cat]}. Right?"])
    reply = f"Yes, {a(member)} is {CATEGORY[cat]}." if truth else f"No, {a(member)} is {CATEGORY[real]}."
    return {"question": question, "answer": "yes" if truth else "no", "answer_type": "yesno", "reply": reply}
