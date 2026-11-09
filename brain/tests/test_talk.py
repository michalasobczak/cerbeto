"""`cerbeto chat`: conversation memory, stop at the next speaker, probing commands — on a tiny student."""
import pytest
import torch

from cerbeto import paths
from cerbeto.model.student import Student, StudentConfig
from cerbeto.talk import Talker

needs_tokenizer = pytest.mark.skipif(not paths.TOKENIZER_FILE.exists(), reason="tokenizer not trained")


@pytest.fixture
def talker(monkeypatch):
    from cerbeto.tokenizer import load
    tok = load()
    scripted = tok.encode(" hi there!<|human|> more").ids

    def fake_generate(self, idx, max_new_tokens, **_):
        return torch.cat([idx, torch.tensor([scripted])], dim=1)

    monkeypatch.setattr(Student, "generate", fake_generate)
    cfg = StudentConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_heads=2, head_dim=16, mlp_hidden=64,
                        n_prelude=1, n_core=1, n_coda=1, max_seq_len=128)
    return Talker(Student(cfg, seed=0), torch.device("cpu"))


@needs_tokenizer
def test_reply_stops_at_next_speaker_and_is_remembered(talker):
    assert talker.reply("Hello!") == "hi there!" and talker.history == ""        # by default every line is separate
    talker.memory = True
    assert talker.reply("Hello!") == "hi there!"
    assert talker.history == "<|human|> Hello!\n<|student|> hi there!<|eos|>"
    talker.reply("How are you?")
    assert talker.history.count("<|human|>") == 2


@needs_tokenizer
def test_probes_do_not_touch_the_conversation(talker):
    assert talker.ask("Is a cat an animal?", 3) == ["hi there!"] * 3
    assert set(talker.think("What color is the sky?")) == {1, 2, 4, 8}
    assert talker.history == ""


@needs_tokenizer
def test_story_runs_past_speaker_tokens_until_eos(talker):
    assert talker.story("Once upon a time").startswith("hi there!")


def test_printer_streams_the_thought_then_the_answer_word_by_word():
    import io
    from cerbeto.talk import Printer
    out = io.StringIO()
    p = Printer(out)
    for raw in ["<|think|> The sun", "<|think|> The sun is hot.<|/think|>", "<|think|> The sun is hot.<|/think|><|answer|> Yes",
                "<|think|> The sun is hot.<|/think|><|answer|> Yes,", "<|think|> The sun is hot.<|/think|><|answer|> Yes, it is."]:
        p(raw)
    p.end(type("T", (), {"last_thought": "The sun is hot."})(), "Yes, it is.")
    assert out.getvalue() == "\033[2m  (thinks: The sun is hot.)\033[0m\nstudent> Yes, it is.\n"


@needs_tokenizer
def test_streamed_reply_is_the_same_reply_token_by_token(talker, monkeypatch):
    tok = talker.tok
    script = tok.encode("<|answer|> Yes, it is.<|eos|> never").ids
    start = {}

    def scripted_forward(self, idx, k=4):        # always puts the next scripted token on top
        n = start.setdefault("n", idx.shape[1])
        logits = torch.zeros(1, idx.shape[1], self.cfg.vocab_size)
        logits[0, -1, script[idx.shape[1] - n]] = 1.0
        return logits

    monkeypatch.setattr(Student, "forward", scripted_forward)
    seen = []
    assert talker.reply("Is the sun hot?", temperature=0.0, on_raw=seen.append) == "Yes, it is."
    assert len(seen) > 2 and all(b.startswith(a) for a, b in zip(seen, seen[1:]))     # the text only grows
    assert "never" not in seen[-1]                                                     # stopped at <|eos|>
