"""Spec 0010: long conversations and relations from code, held-out probes, the shelf mix with a pass limit."""
import random
import re

import numpy as np
import pytest

from cerbeto.train import conversations as c
from cerbeto.train import final_test as f


class FakeTok:
    def encode(self, text):
        return type("E", (), {"ids": text.split()})()


def pairs(seed=0):
    p = f.Pairs.__new__(f.Pairs)
    p.rng, p.mix, p.turns, p.tok = random.Random(seed), f.ARM_PAIR_MIX["bil"], f.ARM_TURNS["bil"], FakeTok()
    story = ("<|human|> Tell me a story about Rex.\n<|student|>", "<|answer|> Rex the dog ran.<|eos|>", "dog", "Rex", 20)
    p.stories, p.named = [story], [story]
    p.lessons = iter(lambda: None, 1)
    p.one_lesson = None
    return p


@pytest.fixture(autouse=True)
def no_lessons(monkeypatch):
    monkeypatch.setattr(f, "lesson_text", lambda _: f.turn("What color is the sky?", "The sky is blue."))


def test_long_dialogues_have_up_to_eight_turns_and_ask_back_at_the_end():
    p = pairs()
    lengths, asked_back = [], 0
    for _ in range(400):
        d = p.dialogue(10_000)
        lengths.append(len(d))
        first, last = d[0][0], d[-1][0]
        if len(d) > 2 and ("named" not in first and re.search(r"called|named|name is|I'm|is in|went to|playing|my new|I have", first)) \
                and ("my" in last.lower() or "Where" in last or "Who is" in last):
            asked_back += 1
    assert 2 <= min(lengths) and max(lengths) == 8
    assert asked_back > 20


def test_remembered_fact_answers_contain_the_fact():
    rng = random.Random(3)
    for _ in range(200):
        (say, ack), qs = c.remembered_fact(rng)
        for q, a, words in qs:
            assert any(w in a.lower() for w in words), (say, q, a)


def test_relations_are_right():
    rng = random.Random(4)
    for _ in range(300):
        q, a, words = c.relation(rng)
        assert any(w in re.findall(r"[a-z]+", a.lower()) for w in words), (q, a)
        scale = next((s for s in c.SCALES if f" {s[0]} than " in q or f" {s[1]} than " in q), None)
        if scale and "Who is the" in q:                         # rebuild the order from the facts
            more, less, most, least = scale
            above = [(x, y) for x, w, y in re.findall(r"(\w+) is (\w+) than (\w+)\.", q) if w == more] + \
                    [(y, x) for x, w, y in re.findall(r"(\w+) is (\w+) than (\w+)\.", q) if w == less]
            people = {p for pair in above for p in pair}
            top = (people - {y for _, y in above}).pop()
            bottom = (people - {x for x, _ in above}).pop()
            assert a.startswith(top if q.endswith(f"{most}?") else bottom), (q, a)


def test_probes_never_use_training_names():
    train = set(c.TRAIN_NAMES + c.TRAIN_PETS + c.TRAIN_PET_NAMES + c.TRAIN_PLACES + c.TRAIN_THINGS)
    for says, _ in c.long_dialogue_probe():
        for s in says:
            assert not (set(re.findall(r"[A-Za-z]+", s)) & train - {"dog", "cat", "bird"}), s
    for q, _ in c.relation_probe():
        assert not (set(re.findall(r"[A-Za-z]+", q)) & train), q
    assert len(c.long_dialogue_probe()) == 30 and len(c.relation_probe()) == 50


def test_shelf_mix_stops_at_the_pass_limit():
    stream = lambda n: type("S", (), {"tokens": np.zeros(n)})()       # noqa: E731
    mix = f.ShelfMix({"a": 0.9, "b": 0.1}, {"a": stream(100), "b": stream(10_000)})
    rng = np.random.default_rng(0)
    for _ in range(2000):
        s = mix.pick(rng)
        mix.took(s, 10)
    assert mix.used["a"] <= f.MAX_PASSES * 100 + 10
    assert mix.passes()["a"] >= f.MAX_PASSES - 0.1
    resumed = f.ShelfMix({"a": 0.9, "b": 0.1}, {"a": stream(100), "b": stream(10_000)}, mix.used)
    assert resumed.used == mix.used
