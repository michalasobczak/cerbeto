"""Nursery exams (spec 0003): BLiMP scoring and the conversation probe, without GPU or a live teacher."""
import pytest
import torch

from cerbeto import paths
from cerbeto.exams import conversation
from cerbeto.model.student import Student, StudentConfig

needs_tokenizer = pytest.mark.skipif(not paths.TOKENIZER_FILE.exists(), reason="tokenizer not trained")


def test_grade_parses_teacher_json(monkeypatch):
    monkeypatch.setattr(conversation, "chat", lambda *a, **k: '{"score": 4, "reason": "fits"}')
    assert conversation.grade("Hello!", "hi!", "rubric") == {"score": 4, "reason": "fits"}


def test_unparseable_grade_counts_as_failure(monkeypatch):
    monkeypatch.setattr(conversation, "chat", lambda *a, **k: "I think it is fine")
    assert conversation.grade("Hello!", "hi!", "rubric")["score"] == 1


@needs_tokenizer
def test_reply_stops_at_the_next_speaker(monkeypatch):
    from cerbeto.tokenizer import load
    tok = load()
    scripted = tok.encode(" hi! <|human|> more").ids

    def fake_generate(self, idx, max_new_tokens, **_):
        return torch.cat([idx, torch.tensor([scripted])], dim=1)

    monkeypatch.setattr(Student, "generate", fake_generate)
    model = Student(StudentConfig(), seed=0)
    assert conversation.reply(model, "Hello!", k=1, device="cpu") == "hi!"


@needs_tokenizer
@pytest.mark.skipif(not (paths.LIBRARY / "blimp").exists(), reason="BLiMP not downloaded")
def test_blimp_prefers_higher_logprob_sentence(monkeypatch):
    from cerbeto.exams import blimp
    monkeypatch.setattr(blimp, "pairs", lambda limit=None: {"toy": [("good one", "bad one")] * 3})
    monkeypatch.setattr(blimp, "sentence_logprobs",
                        lambda m, s, k, d: [-1.0 if x.startswith("good") else -5.0 for x in s])
    assert blimp.evaluate(None, 1, "cpu")["accuracy"] == 1.0
