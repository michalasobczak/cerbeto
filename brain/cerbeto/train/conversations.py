"""Conversation material checked by code (spec 0010): facts the human states and asks back later, two-step relations,
and the held-out probes for long conversations and relations.

Training and the probes use disjoint names, things and places, so a probe answer can only come from the context.
"""
import random

# --- vocab: training and held-out sets never share a name, thing or place ---------------------------------
TRAIN_NAMES = ["Anna", "Tom", "Mia", "Leo", "Emma", "Max", "Lily", "Jack", "Zoe", "Ella", "Lucy",
               "Noah", "Ruby", "Oscar", "Grace", "Finn", "Ivy", "Jake", "Sara", "Luke", "Nora", "Adam", "Kate",
               "Dan", "Eva", "Tim", "Rosa", "Paul", "Molly", "Hugo", "Alice", "Owen", "Chloe", "Ryan"]
PROBE_NAMES = ["Tilda", "Bruno", "Greta", "Felix", "Hazel", "Milo", "Wren", "Otto", "Pia", "Ezra", "Juno", "Kai",
               "Lola", "Nico", "Opal", "Theo"]
TRAIN_PETS = ["dog", "cat", "rabbit", "fish", "bird", "hamster", "turtle", "pony"]
PROBE_PETS = ["parrot", "frog", "goat", "duck"]
TRAIN_PET_NAMES = ["Rex", "Spot", "Bella", "Coco", "Buddy", "Daisy", "Charlie", "Lucky", "Shadow", "Pepper",
                   "Biscuit", "Sunny", "Ginger", "Patch", "Bubbles", "Snowy"]
PROBE_PET_NAMES = ["Pickle", "Waffle", "Noodle", "Muffin", "Pebble", "Tofu", "Clover", "Maple"]
TRAIN_PLACES = ["kitchen", "garden", "park", "school", "bedroom", "library", "shop", "beach"]
PROBE_PLACES = ["barn", "forest", "zoo", "farm"]
TRAIN_THINGS = ["ball", "hat", "kite", "cup", "bike", "book", "car", "bag"]
PROBE_THINGS = ["scarf", "boat", "drum", "lamp"]
COLORS = ["red", "blue", "green", "yellow", "pink", "purple", "orange", "white", "black", "brown"]
# comparisons: (more, less, most, least)
SCALES = [("taller", "shorter", "tallest", "shortest"), ("older", "younger", "oldest", "youngest"),
          ("faster", "slower", "fastest", "slowest"), ("bigger", "smaller", "biggest", "smallest"),
          ("stronger", "weaker", "strongest", "weakest")]
ACKS = ["Okay!", "I'll remember that.", "That's nice!", "Okay, I will remember.", "Got it!", "Oh, nice!"]


def vocab(held_out: bool) -> dict:
    if held_out:
        return {"names": PROBE_NAMES, "pets": PROBE_PETS, "pet_names": PROBE_PET_NAMES, "places": PROBE_PLACES,
                "things": PROBE_THINGS}
    return {"names": TRAIN_NAMES, "pets": TRAIN_PETS, "pet_names": TRAIN_PET_NAMES, "places": TRAIN_PLACES,
            "things": TRAIN_THINGS}


# --- facts the human states, asked back later ---------------------------------------------------------------
def remembered_fact(rng: random.Random, held_out: bool = False) -> tuple[tuple[str, str], list[tuple[str, str, list[str]]]]:
    """(statement turn, [(question, answer, words that must be in the answer)]). The question comes later."""
    v = vocab(held_out)
    kind = rng.randrange(4)
    if kind == 0:
        pet, name = rng.choice(v["pets"]), rng.choice(v["pet_names"])
        say = rng.choice([f"My {pet} is called {name}.", f"I have a {pet}. Its name is {name}.",
                          f"I have a {pet} named {name}."])
        qs = [(f"What is my {pet} called?", f"Your {pet} is called {name}.", [name.lower()]),
              (f"What is the name of my {pet}?", f"Your {pet}'s name is {name}.", [name.lower()]),
              (f"Who is {name}?", f"{name} is your {pet}.", [pet])]
    elif kind == 1:
        color, thing = rng.choice(COLORS), rng.choice(v["things"])
        say = rng.choice([f"I have a {color} {thing}.", f"My {thing} is {color}.", f"Look, my new {thing} is {color}!"])
        qs = [(f"What color is my {thing}?", f"Your {thing} is {color}.", [color])]
    elif kind == 2:
        name, place = rng.choice(v["names"]), rng.choice(v["places"])
        say = rng.choice([f"{name} is in the {place}.", f"{name} went to the {place}.", f"{name} is playing in the {place}."])
        qs = [(f"Where is {name}?", f"{name} is in the {place}.", [place])]
    else:
        name = rng.choice(v["names"])
        say = rng.choice([f"My name is {name}.", f"Hi! I'm {name}.", f"Hello, my name is {name}."])
        ack = f"Hello, {name}!"
        return (say, ack), [("What is my name?", f"Your name is {name}.", [name.lower()]),
                            ("Do you remember my name?", f"Yes, your name is {name}.", [name.lower()])]
    return (say, rng.choice(ACKS)), qs


# --- two facts joined -----------------------------------------------------------------------------------------
def relation(rng: random.Random, held_out: bool = False) -> tuple[str, str, list[str]]:
    """One turn: two or three facts and a question that needs them together. (question, answer, words)."""
    v = vocab(held_out)
    kind = rng.randrange(4)
    if kind <= 1:                                                     # a chain of comparisons
        more, less, most, least = rng.choice(SCALES)
        a, b, c = rng.sample(v["names"], 3)
        facts = f"{a} is {more} than {b}. {b} is {more} than {c}."
        if rng.random() < 0.3:
            facts = f"{b} is {less} than {a}. {c} is {less} than {b}."
        q = rng.randrange(4)
        if q == 0:
            return f"{facts} Who is the {most}?", f"{a} is the {most}.", [a.lower()]
        if q == 1:
            return f"{facts} Who is the {least}?", f"{c} is the {least}.", [c.lower()]
        if q == 2:
            return f"{facts} Is {a} {more} than {c}?", f"Yes. {a} is {more} than {b}, and {b} is {more} than {c}.", ["yes"]
        return f"{facts} Is {c} {more} than {a}?", f"No. {c} is {less} than {b}, and {b} is {less} than {a}.", ["no"]
    if kind == 2:                                                     # owner and name
        name, pet, pet_name = rng.choice(v["names"]), rng.choice(v["pets"]), rng.choice(v["pet_names"])
        facts = f"{name} has a {pet}. The {pet} is called {pet_name}."
        if rng.random() < 0.5:
            return f"{facts} What is the name of {name}'s {pet}?", f"{name}'s {pet} is called {pet_name}.", [pet_name.lower()]
        return f"{facts} Who has {pet_name}?", f"{name} has {pet_name}. {pet_name} is {name}'s {pet}.", [name.lower()]
    a, b = rng.sample(v["names"], 2)                                  # who is with whom, where
    place = rng.choice(v["places"])
    return (f"{a} is in the {place}. {b} is with {a}. Where is {b}?", f"{b} is in the {place}, with {a}.",
            [place])


# --- held-out probes ------------------------------------------------------------------------------------------
SELF_PROBE = [("What are you?", ["student"]), ("Are you a dog?", ["no"]), ("Who are you?", ["cerbeto", "student"])]
FACT_PROBE = [("What color is the sky?", ["blue"]), ("What color is grass?", ["green"]),
              ("What does a cow say?", ["moo"]), ("Is a cat an animal?", ["yes"])]


def long_dialogue_probe(n: int = 30, seed: int = 10_010) -> list[tuple[list[str], list[str]]]:
    """Conversations of 4–6 turns with held-out names: a fact or a story, then other subjects, then a question that
    needs the first turn (or about the student, or a plain fact). The last reply must contain one of the words."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        filler = [q for q, _ in rng.sample(FACT_PROBE + SELF_PROBE, 4)]
        kind = i % 3
        if kind == 0:                                                 # the human said something, asked back later
            (say, _), qs = remembered_fact(rng, held_out=True)
            q, _, words = rng.choice(qs)
            says = [say] + filler[:rng.randint(2, 4)] + [q]
        elif kind == 1:                                               # a story, other subjects, then the story
            pet, name = rng.choice(PROBE_PETS), rng.choice(PROBE_PET_NAMES)
            says = [f"Tell me a story about a {pet} named {name}."] + filler[:rng.randint(2, 3)] + \
                   [rng.choice([f"What was the {pet}'s name?", "Who was the story about?"])]
            words = [name.lower()]
        else:                                                         # a story, then about the student or a fact
            q, words = rng.choice(SELF_PROBE + FACT_PROBE)
            says = [f"Tell me a story about a {rng.choice(PROBE_PETS)}."] + filler[:rng.randint(2, 3)]
            says = [s for s in says if s != q] + [q]
        out.append((says, words))
    return out


def relation_probe(n: int = 50, seed: int = 20_010) -> list[tuple[str, list[str]]]:
    rng = random.Random(seed)
    return [(q, words) for q, _, words in (relation(rng, held_out=True) for _ in range(n))]
