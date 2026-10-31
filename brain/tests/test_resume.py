"""Gate G1 (spec 0003): resuming from a checkpoint gives the same next batch and the same loss."""
import copy

import numpy as np
import torch

from cerbeto.train.loop import Trainer, load_config
from tests.conftest import TINY


def _fake_shelves(root, vocab: int) -> None:
    rng = np.random.default_rng(0)
    for shelf in ("s0_speech", "s0_stories"):
        (root / shelf).mkdir(parents=True)
        for split, n in (("train", 50_000), ("valid", 5_000)):
            rng.integers(0, vocab, n, dtype=np.uint16).tofile(root / shelf / f"{split}.bin")


def _trainer(tmp_path) -> Trainer:
    cfg = copy.deepcopy(load_config())
    cfg.update(total_tokens=100_000, micro_batch=2, eval_batches=1, warmup_steps=2)
    cfg["students"]["growing"] = {"grow": True}
    return Trainer("growing", cfg, shelves_root=tmp_path / "shelves", out_root=tmp_path / "ckpt",
                   run_root=tmp_path / "runs", device=torch.device("cpu"), base_config=TINY)


def test_resume_reproduces_next_batch_and_loss(tmp_path):
    _fake_shelves(tmp_path / "shelves", TINY.vocab_size)
    a = _trainer(tmp_path)
    for _ in range(3):
        a.step()
    a.save()
    rng_state = copy.deepcopy(a.rng.bit_generator.state)
    batch_a, k_a = a.next_batch()
    a.rng.bit_generator.state = rng_state
    loss_a = a.step()

    b = _trainer(tmp_path)
    assert b.resume()
    assert b.state["step"] == 3
    rng_state = copy.deepcopy(b.rng.bit_generator.state)
    batch_b, k_b = b.next_batch()
    b.rng.bit_generator.state = rng_state
    loss_b = b.step()

    assert k_a == k_b and torch.equal(batch_a, batch_b)
    assert abs(loss_a - loss_b) < 1e-5


def test_growth_on_plateau_follows_schedule(tmp_path):
    _fake_shelves(tmp_path / "shelves", TINY.vocab_size)
    t = _trainer(tmp_path)
    t.cfg["growth"]["schedule"] = [[{"op": "grow_depth"}]]
    before = t.model.num_params()
    t.grow()
    assert t.model.cfg.n_core == TINY.n_core + 1 and t.model.num_params() > before
    t.step()  # the transplanted optimizer trains the grown student
    t.grow()  # schedule exhausted → no-op
    assert t.state["growth_index"] == 1
