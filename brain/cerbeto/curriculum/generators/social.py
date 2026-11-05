"""Preschool — social skills: polite routines, agreeing / disagreeing with the truth, opinions, clarification.

"I don't know" is not a separate skill: every skill's pool carries ~10% unanswerable questions (honesty rule).
"""
import random

from cerbeto.curriculum import CATEGORY, a, check_steps, plural, skill, world
from cerbeto.curriculum.generators.common import personal
from cerbeto.curriculum.generators.world_knowledge import CONFUSABLE

STAGE = "preschool"


def _fill(rng: random.Random, line: str) -> str:
    w = world()
    for key, values in {**w["fill"], "name": w["names"], **{k: w["clarify"][k] for k in ("verb", "lverb", "there")}}.items():
        if "{" + key + "}" in line:
            line = line.replace("{" + key + "}", rng.choice(values))
    return line


@skill("social.routines", STAGE, personal)
def routines(rng: random.Random, variant: str, difficulty: int) -> dict:
    kinds = world()["routines"]
    if variant == "ood_combo":      # a greeting first, then the line that needs the answer
        kind = rng.choice(sorted(k for k in kinds if k not in ("greeting", "how_are_you")))
    else:
        kind = rng.choice(sorted(kinds))
    r = kinds[kind]
    if variant == "ood_hard":       # a described situation instead of the line itself
        question = rng.choice(r["situation"])
    elif variant == "ood_combo":
        question = f"{rng.choice(['Hi!', 'Hello!', 'Hi there!'])} {_fill(rng, rng.choice(r['say']))}"
    else:
        question = _fill(rng, rng.choice(r["say"]))
    return {"question": question, "answer": r["accept"][0], "answer_type": "phrase", "accept": r["accept"],
            "reply": r["reply"]}


def _statements(rng: random.Random, variant: str) -> tuple[str, bool, str]:
    """A statement about the world, whether it is true, and the corrected statement."""
    w = world()
    kind = rng.choice(["color", "sound", "category", "ability"])
    if kind == "color":
        thing = rng.choice(sorted(w["colors"]))
        right = w["colors"][thing][0]
        wrong = rng.choice([c for c in w["color_words"] if c not in w["colors"][thing]])
        the = "The" if thing in ("sky", "grass", "snow", "sun", "milk", "coal", "chocolate") else a(thing).split()[0].capitalize()
        truth = rng.random() < 0.5
        return f"{the} {thing} is {right if truth else wrong}.", truth, f"{the} {thing} is {right}."
    if kind == "sound":
        animals = [x for x in sorted(w["sounds"]) if x not in w["sounds_rare"]]
        animal = rng.choice(animals)
        right = w["sounds"][animal][0]
        wrong = rng.choice([w["sounds"][x][0] for x in animals if w["sounds"][x][0] not in w["sounds"][animal]])
        truth = rng.random() < 0.5
        return f"{a(animal).capitalize()} says {right if truth else wrong}.", truth, \
            f"{a(animal).capitalize()} says {right}."
    if kind == "category":
        cats = w["categories"]
        cat = rng.choice(sorted(cats))
        rare = set(w["categories_rare"].get(cat, []))
        truth = rng.random() < 0.5
        if truth:
            member = rng.choice([m for m in cats[cat] if m not in rare])
            return f"{a(member).capitalize()} is {CATEGORY[cat]}.", True, f"{a(member).capitalize()} is {CATEGORY[cat]}."
        other = rng.choice([c for c in sorted(cats) if c != cat and c not in CONFUSABLE.get(cat, ())])
        member = rng.choice([m for m in cats[other] if m not in set(w["categories_rare"].get(other, []))])
        return f"{a(member).capitalize()} is {CATEGORY[cat]}.", False, f"{a(member).capitalize()} is {CATEGORY[other]}."
    ability = rng.choice(sorted(w["abilities"]))
    truth = rng.random() < 0.5
    who = rng.choice(w["abilities"][ability] if truth else w["cannot"][ability])
    subject = plural(who).capitalize()
    if variant == "ood_hard":       # negated statements: "Cats can't fly." is true
        stated = f"{subject} can't {ability}."
        return stated, not truth, f"{subject} {'can' if truth else 'cannot'} {ability}."
    return f"{subject} can {ability}.", truth, f"{subject} {'can' if truth else 'cannot'} {ability}."


@skill("social.agree_disagree", STAGE, personal)
def agree_disagree(rng: random.Random, variant: str, difficulty: int) -> dict:
    if variant == "ood_combo":      # agree or disagree with a statement about a short story
        w = world()
        name, obj = rng.choice(w["names"]), rng.choice(w["objects"])
        color = rng.choice(w["color_words"][:8])
        truth = rng.random() < 0.5
        stated = color if truth else rng.choice([c for c in w["color_words"][:8] if c != color])
        context = f"{name} has {a(color)} {obj}."
        statement, corrected = f"{name}'s {obj} is {stated}.", f"{name}'s {obj} is {color}."
    else:
        statement, truth, corrected = _statements(rng, variant)
        context = ""
    tag = rng.choice(["", " Right?", " Is that true?", " Is that right?"]) if variant != "id" or difficulty > 1 else ""
    reply = "Yes, that's right." if truth else f"No. {corrected}"
    return {"context": context, "question": f"{statement}{tag}".strip(), "answer": "yes" if truth else "no",
            "answer_type": "yesno", "reply": reply,
            "steps": check_steps(statement, corrected, truth)}


@skill("social.opinion", STAGE, personal, graded_by="form")
def opinion(rng: random.Random, variant: str, difficulty: int) -> dict:
    w = world()["opinions"]
    topic = rng.choice(sorted(w))
    likes = rng.random() < 0.5
    reason = w[topic][0 if likes else 1]
    if variant == "ood_hard":
        question = f"Why do you like {topic}?" if likes else f"Why don't you like {topic}?"
        reply = f"Because {reason}."
    elif variant == "ood_combo":
        other = rng.choice([t for t in sorted(w) if t != topic])
        question = f"What do you like more, {topic} or {other}? Why?"
        reply = f"I like {topic} more because {w[topic][0]}."
    else:
        question = rng.choice([f"Do you like {topic}?", f"Do you like {topic}? Why?", f"What do you think about {topic}?"])
        reply = f"{'Yes, I like' if likes else 'No, I do not like'} {topic} because {reason}."
    return {"question": question, "answer": "opinion with a reason", "answer_type": "form", "reply": reply}


@skill("social.clarify", STAGE, personal)
def clarify(rng: random.Random, variant: str, difficulty: int) -> dict:
    c = world()["clarify"]
    if variant in ("ood_hard", "ood_combo"):
        w = world()
        obj = rng.choice(w["objects"])
        c1, c2 = rng.sample(w["color_words"][:8], 2)
        context = f"There is {a(c1)} {obj} and {a(c2)} {obj}."
        question = rng.choice([f"Give me the {obj}.", f"Can you bring me the {obj}?"])
        return {"context": context, "question": question, "answer": "which one", "answer_type": "phrase",
                "accept": ["which one", "which", f"the {c1} one or the {c2} one", "what do you mean"],
                "reply": f"Which one? The {c1} {obj} or the {c2} {obj}?"}
    return {"question": _fill(rng, rng.choice(c["say"])), "answer": c["accept"][0], "answer_type": "phrase",
            "accept": c["accept"], "reply": c["reply"]}
