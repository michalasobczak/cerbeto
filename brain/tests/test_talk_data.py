import random

from cerbeto.train import talk_data


def test_spot_check_questions_are_held_out():
    for q in talk_data.SPOT_QUESTIONS:
        assert talk_data.held_out(q + ("" if q.endswith("?") else "?"))
    assert talk_data.held_out("Where do big fish live?")
    assert talk_data.held_out("Can fish swim?")
    assert not talk_data.held_out("Do fish need water?")
    assert not talk_data.held_out("How many legs does a spider have?")


def test_memory_dialogue_asks_only_what_was_told_or_says_it_does_not_know():
    rng = random.Random(0)
    for _ in range(300):
        turns = talk_data.memory_dialogue(rng, [("Is the sea big?", "Yes, the sea is very big.")])
        told = " ".join(q for q, _ in turns if not q.endswith("?"))      # the statements
        assert 3 <= len(turns) <= 11
        assert not set(told.split()) & {n + "." for n in talk_data.SPOT_PETS}
        for q, a in turns:
            if a.startswith("I don't know"):
                assert not any(w in told for w in ("brother", "years old", "favorite color"))


def test_made_up_words_are_not_probe_words():
    rng = random.Random(1)
    assert all(talk_data.made_up(rng) not in talk_data.PROBE_WORDS for _ in range(500))
    q, a = talk_data.unknown(rng)
    assert q.endswith("?") and "know" in a
