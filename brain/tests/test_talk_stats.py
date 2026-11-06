"""`cerbeto chat` /stats: the inspected forward pass equals the normal one, and the report is complete."""
import pytest
import torch

from cerbeto import paths
from cerbeto.model.student import Attention, Student, StudentConfig

CFG = StudentConfig(vocab_size=8192, d_model=64, n_heads=4, head_dim=16, mlp_hidden=128,
                    n_prelude=1, n_core=1, n_coda=1, max_seq_len=128)


def test_captured_attention_gives_the_same_output():
    model = Student(CFG, seed=0).eval()
    x = torch.randint(0, 8192, (1, 20))
    plain = model(x, 2)
    Attention.capture, Attention.capture_rows = [], slice(15, 19)
    try:
        inspected = model(x, 2)
        att = Attention.capture
    finally:
        Attention.capture, Attention.capture_rows = None, slice(None)
    assert torch.allclose(plain, inspected, atol=1e-4)
    assert len(att) == 1 + 2 + 1 and att[0].shape == (1, 4, 4, 20)
    assert torch.allclose(att[0].sum(-1), torch.ones(1, 4, 4))


@pytest.mark.skipif(not paths.TOKENIZER_FILE.exists(), reason="tokenizer not trained")
def test_stats_report_after_a_reply():
    from cerbeto.talk import Talker
    talker = Talker(Student(CFG, seed=0), "cpu", k=2, temperature=0.0, max_new_tokens=8)
    assert talker.stats() == "(no reply yet)"
    talker.reply("hello")
    talker.reply("who are you?")
    report = talker.stats()
    for part in ("── inference ──", "── tokens", "── attention per block", "core 1 · pass 2", "most attended"):
        assert part in report
    assert talker.last["n_hist"] > 1
