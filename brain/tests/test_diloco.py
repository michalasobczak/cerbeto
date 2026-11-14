"""Spec 0015: DiLoCo — wire format, the outer step, the coordinator's rounds and the worker's sync hook (CPU)."""
import json
import socket
import threading
import types

import numpy as np

import pytest
import torch
from torch.nn.utils import parameters_to_vector

from cerbeto.model.student import Student, StudentConfig
from cerbeto.train import diloco as d
from cerbeto.train import final_test as ft

TINY = StudentConfig(vocab_size=512, d_model=64, n_heads=4, head_dim=16, mlp_hidden=128,
                     n_prelude=1, n_core=1, n_coda=1, max_seq_len=64)


def test_wire_round_trip():
    a, b = socket.socketpair()
    v = torch.randn(1000)
    t = threading.Thread(target=d.send, args=(a, {"kind": "x", "n": 3}, d.to_bytes(v.to(torch.bfloat16))))
    t.start()
    h, payload = d.receive(b)
    t.join()
    assert h == {"kind": "x", "n": 3}
    assert torch.equal(d.from_bytes(payload, torch.bfloat16), v.to(torch.bfloat16))
    assert torch.equal(d.from_bytes(d.to_bytes(v), torch.float32), v)


def test_outer_step_matches_torch_nesterov():
    torch.manual_seed(0)
    theta0 = torch.randn(50)
    ours = d.Outer(theta0.clone(), lr=0.7, momentum=0.9)
    ref = torch.nn.Parameter(theta0.clone())
    sgd = torch.optim.SGD([ref], lr=0.7, momentum=0.9, nesterov=True)
    for _ in range(3):
        grads = [torch.randn(50) * 0.01 for _ in range(4)]
        update = ours.step(grads)
        ref.grad = torch.stack(grads).mean(0)
        sgd.step()
        assert update.dtype == torch.bfloat16
    assert torch.allclose(ours.theta, ref.detach(), atol=1e-3)


def test_one_worker_plain_step_lands_on_its_weights():
    """lr 1, no momentum, one worker: θ moves to the worker's weights (up to bf16 rounding of the move)."""
    theta, local = torch.randn(200), torch.randn(200)
    outer = d.Outer(theta.clone(), lr=1.0, momentum=0.0)
    outer.step([(theta - local).to(torch.bfloat16)])
    assert torch.allclose(outer.theta, local, atol=2e-2)


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "RUN", tmp_path / "runs")
    monkeypatch.setattr(ft, "CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(ft, "start_model", lambda arm, device: Student(TINY, seed=0))
    monkeypatch.setitem(d.OUTER, "every", 2)
    monkeypatch.setitem(d.OUTER, "first_wait", 0.3)

    def make(workers=2, late_after=30.0, tokens=1e9):
        c = d.Coordinator("bil2", "t", tokens, workers, lr=0.7, momentum=0.9, late_after=late_after)
        server = d.serve(c, host="127.0.0.1", port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        made.append(server)
        return c, server.server_address[1]
    made = []
    yield make
    for s in made:
        s.shutdown()
        s.server_close()


class Shelves:
    def __init__(self):
        self.used = {"a": 0, "b": 0}


def worker_sync(link, model, state, shelves, out, i):
    try:
        link(model, None, state, shelves)
    except d.Finished:
        pass
    out[i] = state


def test_round_with_two_workers(coordinator):
    c, port = coordinator(workers=2)
    theta0 = c.theta.clone()
    links, models, states, shelves = [], [], [], []
    for r in range(2):
        link = d.Link("127.0.0.1", port, r)
        link.hello()
        link.tokens_at_sync, link.shelves_at_sync = 0, {}
        m = Student(TINY, seed=0)
        with torch.no_grad():                       # each worker moved somewhere else
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.01 * (r + 1))
        sh = Shelves()
        sh.used["a"] = 100 * (r + 1)
        links.append(link), models.append(m), shelves.append(sh)
        states.append({"step": 2, "tokens": 1000 * (r + 1), "last_loss": {"shelf_loss": 3.0, "pair_loss": 1.0}})
    deltas = [(theta0 - parameters_to_vector(m.parameters()).detach()).to(torch.bfloat16) for m in models]
    out = [None, None]
    ts = [threading.Thread(target=worker_sync, args=(links[i], models[i], states[i], shelves[i], out, i)) for i in range(2)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    ref = d.Outer(theta0.clone(), 0.7, 0.9)
    ref.step(deltas, [1000, 2000])                                      # weighted by the tokens each read
    assert torch.equal(c.theta, ref.theta)
    for i in range(2):
        assert torch.equal(links[i].theta, c.theta)                       # every copy identical
        assert torch.equal(parameters_to_vector(models[i].parameters()).detach(), c.theta)
        assert out[i]["tokens"] == 3000 and out[i]["step"] == 2
        assert shelves[i].used == {"a": 300, "b": 0}
    assert c.state["round"] == 1
    assert (ft.RUN / "t" / "log.jsonl").read_text().count("\n") == 1


def test_late_worker_gets_theta(coordinator):
    c, port = coordinator(workers=2, late_after=0.3)
    a, b = d.Link("127.0.0.1", port, 0), d.Link("127.0.0.1", port, 1)
    a.hello(), b.hello()
    for link in (a, b):
        link.tokens_at_sync, link.shelves_at_sync = 0, {}
    m = Student(TINY, seed=0)
    with torch.no_grad():
        next(m.parameters()).add_(0.05)
    a(m, None, {"step": 2, "tokens": 10}, Shelves())            # alone: the round closes after late_after
    assert c.state["round"] == 1 and c.state["tokens"] == 10
    m2 = Student(TINY, seed=1)
    b(m2, None, {"step": 2, "tokens": 10}, Shelves())           # round 0 is over: b gets θ, its work is dropped
    assert torch.equal(b.theta, c.theta)
    assert torch.equal(parameters_to_vector(m2.parameters()).detach(), c.theta)
    assert c.state["tokens"] == 10
    assert b.state["step"] == 2 and 2 < b.next_sync <= 4                  # only the rest of round 1, not a whole one


def test_weighted_mean():
    outer = d.Outer(torch.zeros(3), lr=1.0, momentum=0.0)
    outer.step([torch.ones(3), torch.zeros(3)], [3, 1])
    assert torch.allclose(outer.theta, torch.full((3,), -0.75))


def test_finish_and_resume(coordinator, tmp_path):
    c, port = coordinator(workers=1, tokens=5)
    link = d.Link("127.0.0.1", port, 0)
    link.hello()
    link.tokens_at_sync, link.shelves_at_sync = 0, {}
    m = Student(TINY, seed=0)
    with pytest.raises(d.Finished):
        link(m, None, {"step": 2, "tokens": 10}, Shelves())
    assert (ft.CKPT / "t" / "final" / "model.safetensors").exists()
    assert (ft.CKPT / "t" / "latest" / "outer.pt").exists()
    again = d.Coordinator("bil2", "t", 5, 1, lr=0.7, momentum=0.9, late_after=1)
    assert again.state["round"] == 1
    assert torch.allclose(again.theta, c.theta)
    assert torch.equal(again.outer.buf, c.outer.buf)


def test_not_a_sync_step_does_nothing(coordinator):
    c, port = coordinator(workers=1)
    link = d.Link("127.0.0.1", port, 0)
    link.hello()
    m = Student(TINY, seed=3)
    before = parameters_to_vector(m.parameters()).detach().clone()
    link(m, None, {"step": 1, "tokens": 1}, Shelves())
    assert torch.equal(parameters_to_vector(m.parameters()).detach(), before)
    assert c.state["round"] == 0


class FakeStream:
    """A shelf of random tokens, as ShelfStream answers."""
    def __init__(self, shelf, split, seq_len, vocab=8192):
        self.seq_len, self.tokens = seq_len, np.zeros(50_000, dtype=np.uint16)

    def batch(self, n, rng, device):
        return torch.randint(0, TINY.vocab_size, (n, self.seq_len + 1), device=device)

    def fixed_batches(self, n, batch_size, device, seed=0):
        g = torch.Generator().manual_seed(seed)
        return [torch.randint(0, TINY.vocab_size, (batch_size, self.seq_len + 1), generator=g).to(device)
                for _ in range(n)]


def test_train_saves_under_its_run_name(tmp_path, monkeypatch):
    """The whole final_test.train loop, a save after every step: everything lands in runs/<name> and
    checkpoints/<name> (2026-11-14: a shelf name, then a parameter name, overwrote the run name at the first save)."""
    monkeypatch.setattr(ft, "RUN", tmp_path / "runs")
    monkeypatch.setattr(ft, "CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(ft, "start_model", lambda arm, device: Student(TINY, seed=0))
    monkeypatch.setattr(ft, "ShelfStream", FakeStream)
    monkeypatch.setattr(ft, "load_tokenizer", lambda vocab: None)
    monkeypatch.setattr(ft, "Pairs", lambda *a, **k: types.SimpleNamespace(batch=lambda n: [None] * n))
    monkeypatch.setattr(ft, "encode_dialogues", lambda d, tok, device, max_len: (
        torch.randint(0, TINY.vocab_size, (len(d), 16)), torch.randint(0, TINY.vocab_size, (len(d), 16))))
    monkeypatch.setitem(ft.CFG, "seq_len", 32)
    monkeypatch.setitem(ft.CFG, "windows", 2)
    monkeypatch.setitem(ft.CFG, "save_minutes", 0)
    ft.train("bil2", 1500, name="t1", device=torch.device("cpu"), bf16=False)
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["t1"]
    assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == ["t1"]
    assert (tmp_path / "ckpt" / "t1" / "final" / "model.safetensors").exists()
    assert json.loads((tmp_path / "runs" / "t1" / "state.json").read_text())["tokens"] >= 1500


def test_thermal_pacer(monkeypatch):
    monkeypatch.setattr(ft.time, "sleep", lambda s: None)
    temps = iter([90] * 5 + [70] * 20)
    pacer = ft.ThermalPacer(85, gpu="1", every=1, read=lambda: next(temps))
    pauses = [pacer() for _ in range(25)]
    assert pauses[0] > 0 and pauses[4] > pauses[0]          # too hot: the pause grows
    assert pauses[-1] == 0.0                                # cool again: back to full speed
    assert max(pauses) <= 5.0


def test_validator_logs_theta_of_the_last_round(coordinator, monkeypatch):
    monkeypatch.setattr(ft, "ShelfStream", FakeStream)
    monkeypatch.setattr(ft, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setitem(ft.CFG, "seq_len", 32)
    c, port = coordinator(workers=1, tokens=5)
    link = d.Link("127.0.0.1", port, 0)
    link.hello()
    link.tokens_at_sync, link.shelves_at_sync = 0, {}
    with pytest.raises(d.Finished):                             # one round, and the run is done
        link(Student(TINY, seed=0), None, {"step": 2, "tokens": 10}, Shelves())
    t = threading.Thread(target=d.validate, args=("bil2", "127.0.0.1", port), kwargs={"poll": 0.05})
    t.start()
    t.join(20)
    assert not t.is_alive()
    rows = [json.loads(x) for x in (ft.RUN / "t" / "log.jsonl").read_text().splitlines()]
    assert rows[-1]["round"] == 1 and set(rows[-1]["valid"]) == set(ft.ARM_VALID["bil2"])
    assert c.state["last_valid"] == rows[-1]["valid"]
