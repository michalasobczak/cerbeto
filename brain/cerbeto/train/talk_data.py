"""Conversation tuning data (spec 0011): questions about the world with short direct answers, conversations that
keep several facts and ask for them later, "I don't know" about made-up words, and replies to feelings.

The world questions are written by the teacher (`generate`, run on the MacBook while the mini trains), the rest by
code. Nothing here may teach the spot check (`scripts/hf/spot_check.py`) or an exam: a teacher question is dropped
when it has every content word of a spot-check question ("Where do big fish live?" ⊇ {fish, live}), equals an
exam prompt, or has an exam hash.

    python -m cerbeto.train.talk_data generate --hours 14      # appends to data/talk/qa_teacher.jsonl, resumable
    python -m cerbeto.train.talk_data stats
    python -m cerbeto.train.talk_data sample
"""
import argparse
import json
import random
import re
import time

from cerbeto import paths
from cerbeto.curriculum import task_hash
from cerbeto.exams.school import exam_hashes
from cerbeto.library.shelves import _norm, exam_prompts
from cerbeto.train import conversations as conv

QA_FILE = paths.DATA / "talk" / "qa_teacher.jsonl"

# --- held out: the spot check (scripts/hf/spot_check.py) -------------------------------------------------------
SPOT_QUESTIONS = ["Is fire hot?", "Is the sun hot?", "Is ice cold?", "Is snow cold?", "The sky is green. Is that right?",
                  "A cat can fly. True or false?", "A fish can swim. True or false?", "Is a dog an animal?",
                  "Is a carrot a fruit or a vegetable?", "Where do fish live?", "What do bees make?",
                  "How many legs does a bird have?", "What color is grass?", "What is the capital of France?",
                  "What do cows give us?", "What is the opposite of hot?",
                  "Water freezes and turns into", "The Moon goes around the", "Bees make", "The heart pumps",
                  "Plants need sunlight and", "A baby cat is called a", "The largest planet in our solar system is"]
SPOT_PETS = {"Tom", "Sunny", "Kiwi", "Rex"}          # pet names of the spot check's memory conversations
STOP = set("a an the is are do does can what where who how many which of in on to and or that right true false "
           "give us make called our your my it be".split())


def _stem(w: str) -> str:
    return w[:-2] if w.endswith("es") and len(w) > 4 else w[:-1] if w.endswith("s") and len(w) > 3 else w


def _content(text: str) -> set[str]:
    return {_stem(w) for w in _norm(text).split() if w not in STOP}


SPOT_CONTENT = [_content(q) for q in SPOT_QUESTIONS]


def held_out(question: str) -> bool:
    c = _content(question)
    return any(s and s <= c for s in SPOT_CONTENT)


# --- teacher: questions about the world ------------------------------------------------------------------------
TOPICS = """pets; farm animals; zoo animals; animals that live in water; birds; insects and bugs; animals in the forest;
animals in cold places; animals in hot deserts; baby animals and their names; what animals eat; animal sounds;
dinosaurs; the human body; our senses; teeth and brushing; staying healthy; sleep; food and fruit; vegetables;
cooking and the kitchen; drinks and milk; bread and grains; sweets and treats; weather; rain and clouds; snow and ice;
wind and storms; the four seasons; day and night; the sun; the moon; stars and planets; space and astronauts;
the earth, land and sea; mountains and volcanoes; rivers and lakes; the ocean and beaches; forests and trees;
flowers and plants; how plants grow; seeds; gardens; rocks and sand; water and its forms; fire and heat;
light and shadows; colors; shapes; numbers and counting; big and small things; time, clocks and calendars;
days of the week and months; birthdays; home and rooms; furniture; clothes; shoes and hats; toys and games;
school and the classroom; books and reading; drawing and painting; music and instruments; sports; playground;
cars, buses and trucks; trains; boats and ships; airplanes; bicycles; roads and traffic safety; jobs people do;
doctors and hospitals; fire fighters and police; shops and money; the post and letters; towns and cities;
villages and the countryside; countries and the world; famous big buildings; family; friends and sharing;
good manners; feelings; helping at home; cleaning and washing; electricity and lamps; telephones and computers;
machines and tools; magnets; materials like wood, metal, glass and paper; hot and cold things; heavy and light things;
opposites; what floats and what sinks; sounds and hearing; smells; rainbows; the sky; holidays and festivals;
camping; the seaside; picnics; farming and food from farms; honey and bees; milk, eggs and cheese; where food comes from;
recycling and keeping nature clean; growing up; babies; old people and grandparents; sleep and dreams""".replace("\n", " ")
TOPICS = [t.strip() for t in TOPICS.split(";") if t.strip()]
ANGLES = ["what they are and what they look like", "where they are and where they live", "what they do and what they can do",
          "how many, how big and how long", "why things happen", "simple true-or-false facts",
          "comparing two things (which one, is X or Y)", "what we use them for", "everyday questions a child asks",
          "yes-or-no questions"]
PROMPT = """Write 25 different questions a curious 5-year-old might ask about: {topic} — {angle}.
Each answer: one or two short, simple, TRUE sentences a kind adult would say to a small child; the first words answer the
question directly (yes/no questions start with "Yes" or "No"; true-or-false questions start with "True" or "False").
Mix kinds: what, where, who, why, how many, which, yes/no, "X. True or false?", "Is X or Y?".
Use only common words. No lists, no names of people.
Return JSON: {{"pairs": [{{"q": "...", "a": "..."}}, ...]}}"""


def _ok(q: str, a: str) -> bool:
    if not (8 <= len(q) <= 120 and 2 <= len(a) <= 200 and q.rstrip().endswith("?")):
        return False
    if not (q.isascii() and a.isascii()) or "\n" in q + a or len(re.findall(r"[.!?](\s|$)", a)) > 3:
        return False
    first = _norm(a).split()[:1]
    if re.search(r"\btrue or false\b", q, re.I) and first not in (["true"], ["false"], ["yes"], ["no"]):
        return False
    return True


def generate(hours: float, teacher: str = "gemma4", seed: int | None = None) -> None:
    from cerbeto.teachers import chat
    QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen = {_norm(json.loads(x)["q"]) for x in QA_FILE.read_text().splitlines()} if QA_FILE.exists() else set()
    prompts, hashes = exam_prompts(), exam_hashes()
    rng = random.Random(seed if seed is not None else time.time_ns())
    end, calls, kept, dropped = time.time() + hours * 3600, 0, 0, {"dup": 0, "held_out": 0, "exam": 0, "form": 0}
    with QA_FILE.open("a") as f:
        while time.time() < end:
            topic, angle = rng.choice(TOPICS), rng.choice(ANGLES)
            try:
                out = chat([{"role": "user", "content": PROMPT.format(topic=topic, angle=angle)}], teacher=teacher,
                           temperature=1.0, json_output=True, timeout=600)
                pairs = json.loads(out).get("pairs", [])
            except Exception as e:                                   # a bad JSON or a timeout: next topic
                print("skip:", type(e).__name__, flush=True)
                continue
            calls += 1
            for p in pairs:
                q, a = str(p.get("q", "")).strip(), str(p.get("a", "")).strip()
                n = _norm(q)
                why = ("form" if not _ok(q, a) else "dup" if n in seen else "held_out" if held_out(q)
                       else "exam" if n in prompts or task_hash(q) in hashes else None)
                if why:
                    dropped[why] += 1
                    continue
                seen.add(n)
                f.write(json.dumps({"q": q, "a": a, "topic": topic, "angle": angle}) + "\n")
                kept += 1
            f.flush()
            if calls % 10 == 0:
                print(f"{time.strftime('%H:%M')} calls {calls} kept {kept} (all {len(seen)}) dropped {dropped}", flush=True)


def load_qa(path=None) -> list[tuple[str, str]]:
    f = path or QA_FILE
    return [(d["q"], d["a"]) for d in map(json.loads, f.read_text().splitlines())] if f.exists() else []


# --- code: several facts kept in a conversation ------------------------------------------------------------------
RELATIVES = ["sister", "brother", "friend", "cousin", "teacher", "grandma", "grandpa"]
FOODS = ["apples", "pizza", "pasta", "bananas", "soup", "pancakes", "rice", "carrots", "ice cream", "cheese"]
PET_NAMES = [n for n in conv.TRAIN_PET_NAMES if n not in SPOT_PETS] + ["Fluffy", "Oreo", "Peanut", "Ziggy", "Honey",
                                                                       "Pippa", "Bolt", "Mochi"]
PETS = conv.TRAIN_PETS
SPOT_NAMES = {"Anna", "Jack", "Zoe", "Leo"}                 # people of the spot check's memory conversations
NAMES = [n for n in conv.TRAIN_NAMES if n not in SPOT_NAMES | SPOT_PETS]
AGES = list(range(3, 12))


def _facts(rng: random.Random) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Facts about the human: (statement, acknowledgement, [(question, answer)]). The names are all different, so
    the human's name, a pet's name and a relative's name can be told apart only by the context."""
    name, rel_name = rng.sample(NAMES, 2)
    pet, pet_name, rel = rng.choice(PETS), rng.choice(PET_NAMES), rng.choice(RELATIVES)
    (color, color2), thing, food, age = rng.sample(conv.COLORS, 2), rng.choice(conv.TRAIN_THINGS), rng.choice(FOODS), rng.choice(AGES)
    place = rng.choice(conv.TRAIN_PLACES)
    pool = [
        (rng.choice([f"My name is {name}.", f"Hi! I'm {name}.", f"Hello, my name is {name}."]), f"Hello, {name}!",
         [("What is my name?", f"Your name is {name}."), ("Do you remember my name?", f"Yes, your name is {name}."),
          ("Who am I?", f"You are {name}.")]),
        (rng.choice([f"I have a {pet} called {pet_name}.", f"My {pet} is called {pet_name}.",
                     f"I have a {pet}. Its name is {pet_name}."]), rng.choice(["What a nice name!", "Oh, nice!", "Okay!"]),
         [(f"What is my {pet} called?", f"Your {pet} is called {pet_name}."),
          (f"What is the name of my {pet}?", f"Your {pet}'s name is {pet_name}."),
          ("What pet do I have?", f"You have a {pet}. Its name is {pet_name}."),
          (f"Who is {pet_name}?", f"{pet_name} is your {pet}.")]),
        (f"My {rel} is called {rel_name}.", rng.choice(["Okay!", "Nice!", "I'll remember that."]),
         [(f"What is my {rel}'s name?", f"Your {rel} is called {rel_name}."), (f"Who is {rel_name}?", f"{rel_name} is your {rel}.")]),
        (rng.choice([f"My favorite color is {color}.", f"I like {color} best."]), f"{color.capitalize()} is a nice color!",
         [("What is my favorite color?", f"Your favorite color is {color}.")]),
        (f"I am {age} years old.", rng.choice([f"{age} is a great age!", "Okay!"]),
         [("How old am I?", f"You are {age} years old.")]),
        (rng.choice([f"I love {food}.", f"My favorite food is {food}."]), rng.choice(["Yummy!", "That sounds good!"]),
         [("What food do I like?", f"You like {food}."), ("What is my favorite food?", f"Your favorite food is {food}.")]),
        (f"I have a {color2} {thing}.", rng.choice(conv.ACKS), [(f"What color is my {thing}?", f"Your {thing} is {color2}.")]),
        (f"Today I went to the {place}.", rng.choice(["Did you have fun?", "That sounds fun!"]),
         [("Where did I go today?", f"You went to the {place}.")]),
    ]
    return rng.sample(pool, rng.randint(2, 4))


def memory_dialogue(rng: random.Random, fillers: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """2–4 facts, 0–3 other questions, then 1–3 questions about the facts (one of them may be about a fact never
    told: "I don't know, you did not tell me")."""
    facts = _facts(rng)
    turns = [(s, ack) for s, ack, _ in facts]
    turns += rng.sample(fillers, rng.randint(0, min(3, len(fillers))))
    asks = [rng.choice(qs) for _, _, qs in rng.sample(facts, rng.randint(1, len(facts)))]
    if rng.random() < 0.15:
        asks.append(rng.choice([("What is my brother's name?", "I don't know. You did not tell me."),
                                ("How old am I?", "I don't know. You did not tell me how old you are."),
                                ("What is my favorite color?", "I don't know. You did not tell me your favorite color.")]))
        told = " ".join(s for s, _ in turns)
        if any(w in told for w in ("brother", "years old", "favorite color", "best")):
            asks.pop()
    rng.shuffle(asks)
    return turns + asks


# --- code: made-up words and feelings ------------------------------------------------------------------------------
SYL = ["bla", "zor", "fip", "mur", "tak", "lom", "quin", "dro", "vex", "plo", "snib", "gru", "tef", "wam", "yop", "kro"]
PROBE_WORDS = {"zorblat", "florp", "blicket", "wug"}          # used by probes: never trained


def made_up(rng: random.Random) -> str:
    while (w := "".join(rng.sample(SYL, rng.randint(2, 3)))) in PROBE_WORDS:
        pass
    return w


def unknown(rng: random.Random) -> tuple[str, str]:
    w = made_up(rng)
    q, a = rng.choice([(f"What is a {w}?", f"I don't know what a {w} is. I have never heard that word."),
                       (f"Do you know what a {w} is?", f"No, I don't know what a {w} is."),
                       (f"Where does a {w} live?", f"I don't know. I don't know what a {w} is."),
                       (f"Can a {w} fly?", f"I don't know. I have never heard of a {w}."),
                       (f"What color is a {w}?", f"I don't know what a {w} is, so I don't know its color.")])
    return q, a


FEELINGS = [
    (["I feel sad today.", "I am sad.", "I'm feeling sad."], ["I'm sorry you feel sad. Do you want to tell me what happened?",
                                                            "Oh no, I'm sorry. I hope you feel better soon."]),
    (["My dog is sick.", "My cat is ill."], ["Oh no, I'm sorry. I hope your pet gets better soon."]),
    (["I lost my toy.", "I can't find my ball."], ["Oh no! Let's think. Where did you last see it?"]),
    (["I am happy today!", "I feel great!"], ["That's wonderful! I'm happy for you.", "Yay! What made you happy?"]),
    (["I am scared of the dark.", "I'm afraid of thunder."], ["It's okay to be scared. You are safe, and grown-ups can help you."]),
    (["I am angry.", "I'm so cross!"], ["It's okay to feel angry. Let's take a deep breath together."]),
    (["I fell and hurt my knee.", "I hurt my hand."], ["Ouch! I'm sorry. Ask a grown-up to look at it."]),
    (["It's my birthday today!", "Today is my birthday!"], ["Happy birthday! I hope you have a wonderful day!"]),
    (["I am bored.", "I have nothing to do."], ["Let's think of a game! You could draw a picture or read a book."]),
    (["I am tired.", "I'm sleepy."], ["Then it's time to rest. Sleep helps you feel better."]),
    (["My friend did not play with me.", "Nobody wants to play with me."], ["I'm sorry. That can feel sad. Maybe you can ask someone else to play."]),
    (["I made a drawing!", "I built a tower!"], ["Well done! I'm sure it looks great.", "That's great! What does it look like?"]),
]


def feeling(rng: random.Random) -> tuple[str, str]:
    says, replies = rng.choice(FEELINGS)
    return rng.choice(says), rng.choice(replies)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["generate", "stats", "sample"])
    p.add_argument("--hours", type=float, default=14)
    p.add_argument("--teacher", default="gemma4")
    a = p.parse_args()
    if a.what == "generate":
        generate(a.hours, a.teacher)
    elif a.what == "stats":
        qa = load_qa()
        print(len(qa), "questions,", sum(len(q) + len(x) for q, x in qa) // 4, "~tokens")
    else:
        rng, qa = random.Random(1), load_qa()
        for q, x in rng.sample(qa, min(10, len(qa))):
            print("Q", q, "| A", x)
        for _ in range(2):
            print("---", *memory_dialogue(rng, qa[:50]), sep="\n  ")
        print("---", unknown(rng), feeling(rng))


if __name__ == "__main__":
    main()
