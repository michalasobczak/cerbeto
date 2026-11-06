"""Spec 0005 G3: an end-to-end lesson with a mock teacher records verified experiences in the hippocampus,
and a sleep on them changes the student; the teacher never sees an exam prompt (N1)."""
import torch

from cerbeto.memory.hippocampus import Hippocampus
from cerbeto.model.student import Student, StudentConfig
from cerbeto.school.day import rephrase_ok, run_day
from cerbeto.school.sleep import run_sleep

CFG = StudentConfig(vocab_size=8192, d_model=64, n_heads=4, head_dim=16, mlp_hidden=128,
                    n_prelude=1, n_core=1, n_coda=1, max_seq_len=128)
DAY = {"lessons": 12, "batch": 6, "rephrase_share": 0.5, "k": 1}
SLEEP = {"epochs": 1, "batch": 6, "replay_windows": 1, "lr": 1e-3, "warmup": 1, "surprise_cap": 3, "ewc_lambda": 0}


def test_g3_lesson_end_to_end_with_a_mock_teacher(tmp_path, monkeypatch):
    seen = []

    def ask(messages, think=False):
        seen.append(messages[-1]["content"])
        return "Steps: I think so.\nAnswer: maybe"      # never verifiable → code explanations are used

    monkeypatch.setattr("cerbeto.school.day.exam_hashes", lambda: set())
    model = Student(CFG, seed=0)
    hippo = Hippocampus(tmp_path / "brain.sqlite")
    stats = run_day(model, "cpu", "t", 1, "preschool", DAY, ask, hippo=hippo, log=lambda m: None)
    assert stats["lessons"] == 12
    exp = hippo.pending("t")
    assert len(exp) == 12 and all(e.verified and e.target.endswith("<|eos|>") for e in exp)
    assert all(e.origin == "code" for e in exp)          # the mock teacher was never trusted
    assert seen                                          # but it was asked (rephrasing / tutoring)

    before = {n: p.clone() for n, p in model.named_parameters()}
    monkeypatch.setattr("cerbeto.school.sleep.ShelfStream", _FakeShelf)
    run_sleep(model, "cpu", exp, [], SLEEP, None, log=lambda m: None)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())


class _FakeShelf:
    def __init__(self, shelf, *a, **k):
        self.shelf = shelf

    def batch(self, n, rng, device):
        return torch.randint(0, 8192, (n, 33))


def test_rephrase_keeps_names_numbers_options_and_meaning():
    assert rephrase_ok("Tom has 3 apples. How many apples?", "Tom has 3 apples. Can you tell me how many?")
    assert not rephrase_ok("Tom has 3 apples. How many apples?", "Tom has 4 apples. How many?")
    assert not rephrase_ok("Was Sam tired or awake?", "How did Sam feel?")
    assert rephrase_ok("Was Sam tired or awake?", "Did Sam feel tired or awake?")
    assert not rephrase_ok("Is it big?", "Is it not big?")
    assert not rephrase_ok("Which is bigger, a cat or a dog?", "Which is bigger?")


def test_sleep_recalls_earlier_days_of_the_stage(tmp_path, monkeypatch):
    from cerbeto.memory.hippocampus import Experience
    hippo = Hippocampus(tmp_path / "b.sqlite")
    old = Experience("t", 1, "preschool", "world.facts", "code", "h", "<|human|> Sky?\n<|student|>",
                     "<|answer|> blue<|eos|>", "red", False, True)
    hippo.add([old])
    assert hippo.memories("t", "preschool") == []          # not slept on yet
    hippo.mark_slept([e.id for e in hippo.pending("t")])
    mem = hippo.memories("t", "preschool")
    assert mem == [("<|human|> Sky?\n<|student|>", "<|answer|> blue<|eos|>", False)]
    seen = []
    import cerbeto.school.sleep as sl
    real = sl.encode_pairs
    monkeypatch.setattr(sl, "encode_pairs", lambda pairs, *a, **k: (seen.append(list(pairs)), real(pairs, *a, **k))[1])
    monkeypatch.setattr(sl, "ShelfStream", _FakeShelf)
    new = [Experience("t", 2, "preschool", "world.sizes", "code", "h2", "<|human|> Big?\n<|student|>",
                      "<|answer|> cow<|eos|>", "cow", True, True)]
    run_sleep(Student(CFG, seed=0), "cpu", new, [], {**SLEEP, "memory_share": 1.0}, None, log=lambda m: None,
              memories=mem)
    assert any(p == mem[0][:2] for batch in seen for p in batch)


def test_rephrase_keeps_knowledge_base_words():
    assert rephrase_ok("A duck says moo. Right?", "Does a duck say moo?")
    assert not rephrase_ok("A duck says moo. Right?", "Does a cow say moo?")
    assert rephrase_ok("Is a pear a fruit?", "Would you say pears are fruit?")
    assert not rephrase_ok("What color is the sky?", "What color is it?")


def test_sleep_keeps_the_weights_with_the_best_validation(monkeypatch):
    from cerbeto.memory.hippocampus import Experience
    monkeypatch.setattr("cerbeto.school.sleep.ShelfStream", _FakeShelf)
    model = Student(CFG, seed=0)
    start = {n: p.clone() for n, p in model.state_dict().items()}
    exp = [Experience("t", 1, "preschool", "world.facts", "code", f"h{i}", "<|human|> Sky?\n<|student|>",
                      "<|answer|> blue<|eos|>", "red", False, True) for i in range(12)]
    scores = iter([0.9] + [0.5] * 10)            # the night only gets worse on validation
    out = run_sleep(model, "cpu", exp, [], {**SLEEP, "epochs": 2, "validate_every": 1}, None, log=lambda m: None,
                    validate=lambda: next(scores))
    assert out["kept_step"] == 0
    assert all(torch.equal(start[n], p) for n, p in model.state_dict().items())


def test_sleep_keeps_the_end_when_the_best_is_within_noise(monkeypatch):
    from cerbeto.memory.hippocampus import Experience
    monkeypatch.setattr("cerbeto.school.sleep.ShelfStream", _FakeShelf)
    exp = [Experience("t", 1, "preschool", "world.facts", "code", f"h{i}", "<|human|> Sky?\n<|student|>",
                      "<|answer|> blue<|eos|>", "red", False, True) for i in range(12)]
    cfg = {**SLEEP, "epochs": 2, "validate_every": 1}
    steps = len(run_sleep(Student(CFG, seed=0), "cpu", exp, [], cfg, None, log=lambda m: None,
                          validate=lambda: 0.5)["validation"]) - 1
    for tolerance, kept in ((0.02, steps), (0.005, 0)):
        scores = iter([0.765] + [0.75] * steps)            # the end is 1.5 pp below the start
        out = run_sleep(Student(CFG, seed=0), "cpu", exp, [], {**cfg, "keep_tolerance": tolerance}, None,
                        log=lambda m: None, validate=lambda: next(scores))
        assert out["kept_step"] == kept


def test_batch_widths_come_in_few_shapes():
    from cerbeto.school.smoke import WIDTH_STEP, encode_pairs
    from cerbeto.tokenizer import load as load_tokenizer
    tok = load_tokenizer()
    widths = set()
    for n in range(1, 40):
        x, y = encode_pairs([("<|human|> " + "word " * n + "\n<|student|>", "<|answer|> yes<|eos|>")], tok, "cpu")
        assert x.shape == y.shape and x.shape[1] % WIDTH_STEP == 0
        assert (y[0] != -100).sum() == len(tok.encode("<|answer|> yes<|eos|>").ids)   # padding is never learned
        widths.add(x.shape[1])
    assert len(widths) <= 2


def test_sleep_counts_the_tokens_it_showed_by_source(monkeypatch):
    from cerbeto.memory.hippocampus import Experience
    monkeypatch.setattr("cerbeto.school.sleep.ShelfStream", _FakeShelf)
    today = [Experience("t", 2, "preschool", "world.sizes", "code", f"h{i}", "<|human|> Big?\n<|student|>",
                        "<|answer|> cow<|eos|>", "cow", True, True) for i in range(6)]
    mem = [("<|human|> Sky?\n<|student|>", "<|answer|> blue<|eos|>", False)]
    out = run_sleep(Student(CFG, seed=0), "cpu", today, [], {**SLEEP, "epochs": 2, "memory_share": 1.0}, None,
                    log=lambda m: None, memories=mem)
    diet = out["diet"]
    assert set(diet) == {"today", "earlier_days", "s0_speech", "s0_stories"}
    assert diet["s0_speech"] == diet["s0_stories"] == 32          # one window of 32 inputs per step, 2 steps
    assert diet["today"] > 0 and diet["earlier_days"] > 0
