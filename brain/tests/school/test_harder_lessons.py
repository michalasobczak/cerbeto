"""Harder lessons (spec 0005): lesson-only mixes of skills, weights from the last exam, unchanged nights."""
import json
from collections import Counter

from cerbeto.curriculum import generate, skills, training_stream, world
from cerbeto.exams.grader import grade
from cerbeto.exams.school import config as exam_config, exam_hashes, passed
from cerbeto.school import scheduler
from cerbeto.school.day import lesson_weights
from cerbeto.school.validation import validation_set

MIXES = [s.name for s in skills("preschool", lessons=True) if s.lesson_only and s.name.startswith("mix.")]
LESSON_ONLY = [s.name for s in skills("preschool", lessons=True) if s.lesson_only]


def test_mixes_are_lesson_only():
    assert len(MIXES) == 7 and "read.facts" in LESSON_ONLY
    assert not set(LESSON_ONLY) & {s.name for s in skills("preschool")}
    assert not set(LESSON_ONLY) & {n for names in exam_config()["stages"].values() for n in names}
    assert not set(LESSON_ONLY) & {t.skill for t in validation_set("preschool")}


def test_lessons_never_repeat_an_exam_prompt():
    exam = exam_hashes()
    for name in LESSON_ONLY:
        # answerable ones; the shared "I don't know" questions are filtered by the stream (gate G4)
        assert not {generate(name, "train", seed, unanswerable=False).hash for seed in range(1000)} & exam


def test_reading_tells_the_fact_it_asks_about():
    for seed in range(300):
        t = generate("read.facts", "train", seed, unanswerable=False)
        assert t.context and grade(t, t.reply)
        if t.answer_type == "text" and "opposite" in t.question:
            assert t.answer in t.context.lower()


def test_passing_needs_mastery_and_mean_transfer():
    def stage(*cells):
        return {f"s{i}": {"id": {"accuracy": a, "graded_by": "code"}, "ood_hard": {"accuracy": h}}
                for i, (a, h) in enumerate(cells)} | {"_passed": None}
    rule = {"id": 0.85, "ood_hard": 0.70, "ood_hard_floor": 0.50}
    assert passed(stage((0.9, 0.9), (0.9, 0.55)), rule)              # mean 72.5%, floor kept
    assert not passed(stage((0.9, 0.9), (0.84, 0.9)), rule)          # one skill not mastered
    assert not passed(stage((0.9, 0.95), (0.9, 0.45)), rule)         # one skill without transfer
    assert not passed(stage((0.9, 0.7), (0.9, 0.6)), rule)           # mean 65%


def test_mix_reference_replies_grade_right_and_skip_rare_items():
    w = world()                   # rare items are rare per relation: a bee is a rare animal but not a rare size
    opposites = {x for p in w["opposites_rare"] for x in p}
    rare_of = {"mix.size_opposite": set(w["sizes_rare"]) | opposites, "mix.opposite_statement": opposites,
               "mix.sound_size": set(w["sounds_rare"]) | set(w["sizes_rare"]),
               "mix.clarify_kind": {m for v in w["categories_rare"].values() for m in v},
               "mix.which_category": {m for v in w["categories_rare"].values() for m in v},
               "mix.say_back": set(), "mix.not_statement": set()}
    for name in MIXES:
        rare = rare_of[name]
        for seed in range(300):
            t = generate(name, "train", seed, unanswerable=False)
            assert grade(t, t.reply), (t.prompt, t.reply)
            words = set(t.prompt.lower().replace("?", " ").replace(".", " ").split())
            assert not words & rare, t.prompt


def test_clarify_kind_wrong_reply_is_wrong():
    for seed in range(100):
        t = generate("mix.clarify_kind", "train", seed, unanswerable=False)
        if t.answer_type == "text":          # one thing fits: asking "which one?" is wrong
            assert not grade(t, f"Which one? The {t.accept[0]} or the {t.reject[1]}?")
        else:                                # two things fit: just picking one is wrong
            assert not grade(t, "Thank you!")


def test_weights_follow_the_exam():
    results = {"world.facts": {v: {"accuracy": 0.9, "graded_by": "code"} for v in ("id", "ood_hard", "ood_combo")},
               "social.clarify": {v: {"accuracy": 0.1, "graded_by": "code"} for v in ("id", "ood_hard", "ood_combo")}}
    w = lesson_weights(results, {"focus": 2.0, "mix_weight": 0.5})
    assert abs(w["world.facts"] - 1.2) < 1e-9 and abs(w["social.clarify"] - 2.8) < 1e-9
    assert w["world.sizes"] == 1.0 and all(w[m] == 0.5 for m in LESSON_ONLY)
    stream = training_stream(seed=1, stage="preschool", weights=w, lessons=True)
    counts = Counter(next(stream).skill for _ in range(3000))
    assert counts["social.clarify"] > 1.8 * counts["world.facts"]
    assert sum(counts[m] for m in LESSON_ONLY) > 0


def test_unweighted_stream_has_no_mixes():
    stream = training_stream(seed=3, stage="preschool")
    assert not {next(stream).skill for _ in range(500)} & set(LESSON_ONLY)


def test_unchanged_night_is_no_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "RUN", tmp_path)
    monkeypatch.setattr(scheduler, "STATE", tmp_path / "state.json")
    school = scheduler.School.__new__(scheduler.School)
    school.cfg = {"exam": {"plateau_cycles": 3}}
    school.log_file = tmp_path / "log.txt"
    school.state = {"stage": "preschool", "active": "v0015", "day": 16, "phase": "sleep", "candidate": None,
                    "plateau": 1}
    school.unchanged({"kept_step": 0, "validation": [(0, 0.765), (35, 0.734)]})
    st = school.state
    assert (st["active"], st["day"], st["phase"], st["plateau"]) == ("v0015", 17, "day", 2)
    record = json.loads((tmp_path / "cycles.jsonl").read_text())
    assert record["candidate"] is None and record["promoted"] is False
    assert "76.5%" in school.log_file.read_text()
