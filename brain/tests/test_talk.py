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
