"""School curriculum (spec 0004): task generators whose answer is known to the code.

Every task is generated from `(skill, variant, seed, difficulty)` alone, so it is reproducible (gate G2).
Variants: `train` (difficulty 1–5, the lessons), `id` (exam: same kinds and difficulty as training),
`ood_hard` (exam: beyond the training difficulty) and `ood_combo` (exam: new combinations of known elements).
Exam seeds come from a reserved range that the training stream refuses to use (gate N1).
"""
import hashlib
import random
import re
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import yaml

WORLD_FILE = Path(__file__).with_name("world.yaml")
EXAM_SEED_BASE = 10**9            # seeds ≥ this belong to exam pools only
VARIANTS = ("train", "id", "ood_hard", "ood_combo")
EXAM_VARIANTS = ("id", "ood_hard", "ood_combo")
MAX_TRAIN_DIFFICULTY = 5
UNANSWERABLE_RATE = 0.10          # honesty rule: ~10% of every pool cannot be answered


class ExamSeedError(ValueError):
    """A training generator was asked for a task from the reserved exam range (gate N1)."""


@dataclass
class Task:
    skill: str
    stage: str
    variant: str
    seed: int
    difficulty: int
    question: str
    answer: str                    # canonical answer ("no", "7", "eyes", "I don't know")
    answer_type: str               # yesno | number | text | phrase | unknown | form
    reply: str                     # a natural reply in the student's voice — the lesson target
    context: str = ""
    accept: list[str] = field(default_factory=list)   # all fitting answers (text / phrase)
    reject: list[str] = field(default_factory=list)   # competitors: their presence makes the reply wrong
    steps: list[str] = field(default_factory=list)    # reference solution, step by step
    graded_by: str = "code"        # code | form (soft skills: reported, not used for passing)

    @property
    def prompt(self) -> str:
        return f"{self.context} {self.question}".strip()

    @property
    def hash(self) -> str:
        return task_hash(self.prompt)

    def to_dict(self) -> dict:
        return {**asdict(self), "hash": self.hash}

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def task_hash(prompt: str) -> str:
    norm = re.sub(r"\s+", " ", prompt.lower()).strip()
    return hashlib.blake2b(norm.encode(), digest_size=12).hexdigest()


@cache
def world() -> dict:
    return yaml.safe_load(WORLD_FILE.read_text())


def a(noun: str) -> str:
    return f"an {noun}" if noun[0] in "aeiou" else f"a {noun}"


CATEGORY = {"animal": "an animal", "fruit": "a fruit", "vehicle": "a vehicle", "toy": "a toy",
            "clothes": "a piece of clothing"}


def plural(noun: str) -> str:
    irregular = {"mouse": "mice", "sheep": "sheep", "fish": "fish", "foot": "feet", "tooth": "teeth",
                 "child": "children", "puppy": "puppies", "cherry": "cherries", "strawberry": "strawberries",
                 "bus": "buses", "peach": "peaches", "box": "boxes", "dress": "dresses", "glass": "glasses"}
    return irregular.get(noun, noun + "s")


def check_steps(said: str, fact: str, truth: bool) -> list[str]:
    """Thinking for a yes/no about a statement: repeat the claim, state the fact, then decide — so "yes" or "no"
    comes from comparing the two, not before the fact (the student had learned "No. <the true fact>")."""
    said = said.strip().rstrip(".?!")
    return [f"The question says: {said[0].lower() + said[1:]}.", fact,
            f"So that is {'right' if truth else 'wrong'}, and the answer is {'yes' if truth else 'no'}."]


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Skill:
    name: str
    stage: str
    make: object          # (rng, variant, difficulty) -> Task fields (dict)
    unanswerable: object  # (rng, variant, difficulty) -> Task fields (dict)
    graded_by: str = "code"
    lesson_only: bool = False   # a mix of skills taught in lessons; never in exams or validation


SKILLS: dict[str, Skill] = {}


def skill(name: str, stage: str, unanswerable, graded_by: str = "code", lesson_only: bool = False):
    def register(fn):
        SKILLS[name] = Skill(name, stage, fn, unanswerable, graded_by, lesson_only)
        return fn
    return register


def stages() -> list[str]:
    from cerbeto.curriculum import generators  # noqa: F401 — registers the skills
    return list(dict.fromkeys(s.stage for s in SKILLS.values()))


def skills(stage: str | None = None, lessons: bool = False) -> list[Skill]:
    """Skills of a stage (all stages for None). `lessons=True` adds the lesson-only mixes of skills."""
    from cerbeto.curriculum import generators  # noqa: F401
    return [s for s in SKILLS.values() if (stage is None or s.stage == stage) and (lessons or not s.lesson_only)]


def generate(skill_name: str, variant: str, seed: int, difficulty: int | None = None,
             unanswerable: bool | None = None) -> Task:
    """One task. `unanswerable=None` lets the seed decide (~10%); exam pools set it explicitly."""
    from cerbeto.curriculum import generators  # noqa: F401
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if variant == "train" and seed >= EXAM_SEED_BASE:
        raise ExamSeedError(f"seed {seed} is reserved for exams")
    sk = SKILLS[skill_name]
    rng = random.Random(f"{skill_name}/{variant}/{seed}")
    if difficulty is None:
        difficulty = rng.randint(1, 3) if variant == "id" else (6 if variant == "ood_hard" else rng.randint(1, 5))
    if variant == "train":
        difficulty = min(difficulty, MAX_TRAIN_DIFFICULTY)
    if unanswerable is None:
        unanswerable = rng.random() < UNANSWERABLE_RATE
    fields = (sk.unanswerable if unanswerable else sk.make)(rng, variant, difficulty)
    if unanswerable:
        fields.setdefault("answer", "I don't know")
        fields.setdefault("answer_type", "unknown")
        fields.setdefault("reply", "I don't know.")
    return Task(skill=skill_name, stage=sk.stage, variant=variant, seed=seed, difficulty=difficulty,
                graded_by="code" if unanswerable else sk.graded_by, **fields)


def training_stream(seed: int = 0, stage: str | None = None, exclude: set[str] | None = None,
                    weights: dict[str, float] | None = None, lessons: bool = False):
    """Endless lessons: random skills of the stage(s), training seeds only, never an exam prompt (gate G4).

    `weights` (skill → weight, default 1) makes some skills more frequent; `lessons=True` adds the mixes of skills."""
    rng = random.Random(seed)
    names = [s.name for s in skills(stage, lessons)]
    w = [max(0.0, (weights or {}).get(n, 1.0)) for n in names]
    exclude = exclude or set()
    while True:
        name = rng.choices(names, w)[0] if weights else rng.choice(names)   # unweighted: the same stream as before
        task = generate(name, "train", rng.randrange(EXAM_SEED_BASE))
        if task.hash not in exclude:
            yield task


@cache
def knowledge_words() -> frozenset[str]:
    """Words of the knowledge base a question is about (things, animals, colours, sounds, categories, opposites):
    a rephrasing must keep every one of them; an answer that repeats all of them affirms the statement."""
    w = world()
    words = set(w["colors"]) | set(w["color_words"]) | set(w["sounds"]) | set(w["sizes"]) | set(w["objects"])
    words |= {s for v in w["sounds"].values() for s in v} | set(w["categories"])
    words |= {m for v in w["categories"].values() for m in v} | {x for p in w["opposites"] for x in p}
    words |= set(w["abilities"]) | {m for v in w["abilities"].values() for m in v} | set(w["places"])
    return frozenset(x for x in words if x.isalpha() and len(x) > 2)
