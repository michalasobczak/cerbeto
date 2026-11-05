"""Preschool mixes — lessons that combine two known skills in one question (spec 0005, harder lessons).

After SKILL-MIX (Yu et al. 2023): a small model that practises *some* combinations of skills handles *other*
combinations better. These combinations are lesson-only: they are never in an exam or in validation, and none of them
is an exam form — the exam's `ood_combo` forms (opposite of the opposite, which crayon, which animal says, …) stay
held out, so the exam still measures transfer to combinations the student has not practised. Rare items (`*_rare`
in world.yaml) are never used: they stay the exam's test of knowledge the lessons never gave.
"""
import random

from cerbeto.curriculum import a, skill, world
from cerbeto.curriculum.generators.common import personal
from cerbeto.curriculum.generators.world_knowledge import CONFUSABLE, opposites_of

STAGE = "preschool"
TAGS = [" Right?", " Is that true?", " Is that right?"]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _opposite_words() -> list[str]:
    w = world()
    rare = {x for p in w["opposites_rare"] for x in p}
    return sorted({x for p in w["opposites"] for x in p[:2] if x not in rare and x not in ("yes", "no")})


def _two_sizes(rng: random.Random, pool: list[str]) -> tuple[str, str]:
    sz = world()["sizes"]
    while True:
        x, y = rng.sample(pool, 2)
        if abs(sz[x] - sz[y]) >= 2:
            return x, y


@skill("mix.size_opposite", STAGE, personal, lesson_only=True)
def size_opposite(rng: random.Random, variant: str, difficulty: int) -> dict:
    """opposites ∘ sizes: "What is the opposite of small? Which one is bigger, a cat or a horse?" """
    w = world()
    sz = w["sizes"]
    pool = [x for x in sorted(sz) if x not in w["sizes_rare"]]
    word = rng.choice(["big", "small"])
    opp = opposites_of(word)[0]
    more = "bigger" if opp == "big" else "smaller"
    x, y = _two_sizes(rng, pool)
    pick = (max if opp == "big" else min)((x, y), key=sz.get)
    other = y if pick == x else x
    return {"question": f"What is the opposite of {word}? Which one is {more}, {a(x)} or {a(y)}?",
            "answer": pick, "answer_type": "text", "accept": [pick], "reject": [other],
            "reply": f"The opposite of {word} is {opp}. {_cap(a(pick))} is {more}.",
            "steps": [f"The opposite of {word} is {opp}.", f"{_cap(a(pick))} is {more} than {a(other)}."]}


@skill("mix.opposite_statement", STAGE, personal, lesson_only=True)
def opposite_statement(rng: random.Random, variant: str, difficulty: int) -> dict:
    """opposites ∘ agree/disagree: "The opposite of hot is cold. Right?" """
    words = _opposite_words()
    word = rng.choice(words)
    right = opposites_of(word)
    truth = rng.random() < 0.5
    stated = right[0] if truth else rng.choice([x for x in words if x != word and x not in right])
    reply = "Yes, that's right." if truth else f"No. The opposite of {word} is {right[0]}."
    return {"question": f"The opposite of {word} is {stated}.{rng.choice(TAGS)}", "answer": "yes" if truth else "no",
            "answer_type": "yesno", "reply": reply,
            "steps": [f"The opposite of {word} is {right[0]}. So the answer is {'yes' if truth else 'no'}."]}


@skill("mix.sound_size", STAGE, personal, lesson_only=True)
def sound_size(rng: random.Random, variant: str, difficulty: int) -> dict:
    """facts (sounds) ∘ sizes: "Which is bigger, the one that says moo or the one that says squeak?" """
    w = world()
    sz, sounds = w["sizes"], w["sounds"]
    first = [sounds[x][0] for x in sounds]
    pool = sorted(x for x in sounds if x in sz and x not in w["sounds_rare"] and x not in w["sizes_rare"]
                  and first.count(sounds[x][0]) == 1)
    x, y = _two_sizes(rng, pool)
    big = rng.random() < 0.5
    more = "bigger" if big else "smaller"
    pick = (max if big else min)((x, y), key=sz.get)
    other = y if pick == x else x
    return {"question": f"Which is {more}, the one that says {sounds[x][0]} or the one that says {sounds[y][0]}?",
            "answer": pick, "answer_type": "text", "accept": [pick], "reject": [other],
            "reply": f"{_cap(a(pick))} is {more}. {_cap(a(pick))} says {sounds[pick][0]}.",
            "steps": [f"The one that says {sounds[x][0]} is {a(x)}.", f"The one that says {sounds[y][0]} is {a(y)}.",
                      f"{_cap(a(pick))} is {more} than {a(other)}."]}


KIND = {"animal": "animal", "fruit": "fruit", "vehicle": "vehicle", "toy": "toy"}


@skill("mix.clarify_kind", STAGE, personal, lesson_only=True)
def clarify_kind(rng: random.Random, variant: str, difficulty: int) -> dict:
    """categories ∘ clarify: two things fit the request → "Which one?"; one thing fits → give it."""
    w = world()
    cats = {c: [m for m in w["categories"][c] if m not in w["categories_rare"].get(c, []) and " " not in m
                and not any(m in w["categories"][o] for o in w["categories"] if o != c)] for c in KIND}
    cat = rng.choice(sorted(KIND))
    request = rng.choice([f"Give me the {KIND[cat]}.", f"Can you bring me the {KIND[cat]}?"])
    if rng.random() < 0.5:                     # ambiguous: both fit
        x, y = rng.sample(cats[cat], 2)
        return {"context": f"There is {a(x)} and {a(y)}.", "question": request, "answer": "which one",
                "answer_type": "phrase", "accept": ["which one", "which", f"the {x} or the {y}"],
                "reply": f"Which one? The {x} or the {y}?",
                "steps": [f"The {x} and the {y} are both {a(KIND[cat])}, so I ask which one."]}
    other = rng.choice([c for c in sorted(KIND) if c != cat and c not in CONFUSABLE.get(cat, ())
                        and cat not in CONFUSABLE.get(c, ())])
    x, y = rng.choice(cats[cat]), rng.choice(cats[other])
    things = [x, y]
    rng.shuffle(things)
    return {"context": f"There is {a(things[0])} and {a(things[1])}.", "question": request, "answer": x,
            "answer_type": "text", "accept": [x], "reject": ["which", y],
            "reply": f"Here is the {x}.", "steps": [f"Only the {x} is {a(KIND[cat])}."]}


# ── reading: what the lessons never ask about is told as a fact, and the question is asked about the text ─────

def _size_word(rank: int) -> str:
    return "very small" if rank <= 2 else "small" if rank <= 4 else "big" if rank <= 8 else "very big"


@skill("read.facts", STAGE, personal, lesson_only=True)
def read_facts(rng: random.Random, variant: str, difficulty: int) -> dict:
    """A fact about a rare item in a short text, then a question about it. Rare items are never *asked about*
    without the text in lessons, so the exam's `ood_hard` asks from memory what the student only read."""
    w = world()
    kind = rng.choice(["opposite", "sound", "size", "category"])
    if kind == "opposite":
        pair = rng.choice(w["opposites_rare"])
        word, opp = rng.sample(pair, 2)
        fact = rng.choice([f"{_cap(pair[0])} and {pair[1]} are opposites.", f"The opposite of {word} is {opp}."])
        return {"context": fact, "question": f"What is the opposite of {word}?", "answer": opp, "answer_type": "text",
                "accept": [opp], "reply": f"The opposite of {word} is {opp}."}
    if kind == "sound":
        animal = rng.choice(w["sounds_rare"])
        s = w["sounds"][animal][0]
        fact = rng.choice([f"{_cap(a(animal))} says {s}.", f"Listen! The {animal} goes {s}."])
        return {"context": fact, "question": f"What does {a(animal)} say?", "answer": s, "answer_type": "text",
                "accept": w["sounds"][animal], "reply": f"{_cap(a(animal))} says {s}."}
    if kind == "size":
        sz = w["sizes"]
        item = rng.choice(w["sizes_rare"])
        other = rng.choice([x for x in sorted(sz) if x not in w["sizes_rare"] and abs(sz[x] - sz[item]) >= 2])
        big = rng.random() < 0.5
        pick = (max if big else min)((item, other), key=sz.get)
        more = "bigger" if big else "smaller"
        options = [item, other]
        rng.shuffle(options)
        return {"context": f"{_cap(a(item))} is {_size_word(sz[item])}.",
                "question": f"Which is {more}, {a(options[0])} or {a(options[1])}?", "answer": pick,
                "answer_type": "text", "accept": [pick], "reject": [x for x in options if x != pick],
                "reply": f"{_cap(a(pick))} is {more}."}
    cats = w["categories_rare"]
    cat = rng.choice(sorted(cats))
    member = rng.choice(cats[cat])
    asked = cat if rng.random() < 0.5 else rng.choice([c for c in sorted(w["categories"]) if c != cat
                                                        and c not in CONFUSABLE.get(cat, ()) and cat not in CONFUSABLE.get(c, ())])
    from cerbeto.curriculum import CATEGORY
    truth = asked == cat
    return {"context": f"{_cap(a(member))} is {CATEGORY[cat]}.", "question": f"Is {a(member)} {CATEGORY[asked]}?",
            "answer": "yes" if truth else "no", "answer_type": "yesno",
            "reply": f"Yes, {a(member)} is {CATEGORY[cat]}." if truth else f"No, {a(member)} is {CATEGORY[cat]}."}


# ── forms the exam holds out, practised in a neighbouring form ────────────────────────────────────────────────

@skill("mix.say_back", STAGE, personal, lesson_only=True)
def say_back(rng: random.Random, variant: str, difficulty: int) -> dict:
    """routines in reported speech: 'Tom says: "Here is a ball for you." What do you say?' (the exam's `ood_hard`
    describes a situation instead)."""
    from cerbeto.curriculum.generators.social import _fill
    w = world()
    r = w["routines"][rng.choice(sorted(w["routines"]))]
    name = rng.choice(w["names"])
    line = _fill(rng, rng.choice(r["say"]))
    return {"question": f'{name} says: "{line}" What do you say?', "answer": r["accept"][0], "answer_type": "phrase",
            "accept": r["accept"], "reply": r["reply"]}


@skill("mix.not_statement", STAGE, personal, lesson_only=True)
def not_statement(rng: random.Random, variant: str, difficulty: int) -> dict:
    """agree/disagree with 'is not' about colours (the exam's `ood_hard` negates abilities: "Cats can't fly.")."""
    w = world()
    thing = rng.choice(sorted(w["colors"]))
    right = w["colors"][thing]
    the = "The" if thing in ("sky", "grass", "snow", "sun", "milk", "coal", "chocolate") else a(thing).split()[0].capitalize()
    truth = rng.random() < 0.5                 # truth of "X is not C"
    color = rng.choice([c for c in w["color_words"] if c not in right]) if truth else right[0]
    reply = f"Yes, that's right. {the} {thing} is {right[0]}." if truth else f"No. {the} {thing} is {right[0]}."
    return {"question": f"{the} {thing} is not {color}.{rng.choice(TAGS)}", "answer": "yes" if truth else "no",
            "answer_type": "yesno", "reply": reply,
            "steps": [f"{the} {thing} is {right[0]}. So the answer is {'yes' if truth else 'no'}."]}


@skill("mix.which_category", STAGE, personal, lesson_only=True)
def which_category(rng: random.Random, variant: str, difficulty: int) -> dict:
    """categories as a choice: "Which one is a fruit, a lemon or a car?" (the exam asks the colour this way)."""
    from cerbeto.curriculum import CATEGORY
    w = world()
    plain = {c: [m for m in w["categories"][c] if m not in w["categories_rare"].get(c, [])] for c in w["categories"]}
    cat = rng.choice(sorted(plain))
    other = rng.choice([c for c in sorted(plain) if c != cat and c not in CONFUSABLE.get(cat, ())
                        and cat not in CONFUSABLE.get(c, ())])
    x, y = rng.choice(plain[cat]), rng.choice(plain[other])
    options = [x, y]
    rng.shuffle(options)
    return {"question": f"Which one is {CATEGORY[cat]}, {a(options[0])} or {a(options[1])}?", "answer": x,
            "answer_type": "text", "accept": [x], "reject": [y], "reply": f"{_cap(a(x))} is {CATEGORY[cat]}."}
