"""Gates G2 and G3 (spec 0002): growth preserves behaviour, parameters follow the formula, the limit holds."""
import pytest
import torch

from cerbeto.model import growth
from cerbeto.model.student import Student, StudentConfig, count_params

OPS = [
    ("grow_depth", {}),
    ("grow_width", {"extra_mlp": 32}),
    ("grow_width", {"extra_heads": 2}),
    ("grow_model_dim", {"new_d": 96}),
]


@pytest.mark.parametrize("op,kwargs", OPS)
def test_growth_preserves_function(trained_tiny, op, kwargs):
    model, _, data = trained_tiny
    new, event = getattr(growth, op)(model, **kwargs)
    dlogits, dloss = growth.behaviour_delta(model, new, data, k=3)
    assert dlogits < 1e-4
    assert dloss <= 0.01
    assert event["params_after"] > event["params_before"]


@pytest.mark.parametrize("op,kwargs", OPS)
def test_parameter_count_matches_formula(trained_tiny, op, kwargs):
    model, _, _ = trained_tiny
    new, _ = getattr(growth, op)(model, **kwargs)
    assert new.num_params() == count_params(new.cfg)


@pytest.mark.parametrize("op,kwargs", OPS)
def test_new_units_receive_gradient(trained_tiny, op, kwargs):
    model, _, data = trained_tiny
    new, _ = getattr(growth, op)(model, **kwargs)
    new.train()
    new.loss(data[:, :-1], data[:, 1:], k=2).backward()
    block = new.core[-1]
    writer = {"grow_depth": block.mlp.down.weight.grad,
              "grow_width": block.mlp.down.weight.grad[:, -1] if "extra_mlp" in kwargs else block.attn.o.weight.grad[:, -1],
              "grow_model_dim": block.mlp.down.weight.grad[-1]}[op]
    assert writer.abs().sum() > 0


def test_limit_rejects_growth_and_logs(trained_tiny, tmp_path):
    model, _, data = trained_tiny
    log = tmp_path / "growth_log.jsonl"
    with pytest.raises(growth.GrowthRejected):
        growth.apply(model, "grow_depth", data, k=2, max_params=model.num_params(), max_delta_loss=0.01, log_path=log)
    assert '"accepted": false' in log.read_text()


def test_30m_limit_on_real_config():
    cfg = StudentConfig(d_model=512, n_heads=16, mlp_hidden=1408, n_core=4)
    assert count_params(cfg) <= 30_000_000
    assert count_params(cfg.with_(n_core=5)) > 30_000_000


def test_start_size_within_spec():
    assert count_params(StudentConfig()) <= 10_000_000


def test_optimizer_transplant_keeps_moments(trained_tiny):
    model, opt, data = trained_tiny
    new, _ = growth.grow_width(model, extra_mlp=32)
    opt2 = growth.transplant_optimizer(opt, model, new)
    old_m = opt.state[model.core[0].mlp.gate.weight]["exp_avg"]
    new_m = opt2.state[new.core[0].mlp.gate.weight]["exp_avg"]
    assert torch.equal(new_m[:old_m.shape[0]], old_m)
    assert new_m[old_m.shape[0]:].abs().sum() == 0


def test_renewal_replaces_least_useful_neurons_with_silent_fresh_ones(trained_tiny):
    model, opt, data = trained_tiny
    tracker = growth.NeuronTracker(model)
    model.train()
    model.loss(data[:, :-1], data[:, 1:], k=2)
    tracker.step()
    loss_before = model.loss(data[:, :-1], data[:, 1:], k=2).item()
    renewed = tracker.renew(fraction=0.05, maturity=1, std=0.01, optimizer=opt)
    assert renewed == int(0.05 * 128 * model.cfg.n_blocks)
    fresh = [(b, i) for b, age in enumerate(tracker.age) for i in torch.nonzero(age == 0).flatten().tolist()]
    blocks = list(model.blocks())
    assert len(fresh) == renewed
    assert all(blocks[b].mlp.down.weight[:, i].abs().sum() == 0 for b, i in fresh)
    assert abs(model.loss(data[:, :-1], data[:, 1:], k=2).item() - loss_before) <= 0.01


def test_pruning_removes_persistently_useless_neurons(trained_tiny):
    model, _, data = trained_tiny
    tracker = growth.NeuronTracker(model)
    for block in model.blocks():  # silence neurons 0..9 in every block
        block.mlp.down.weight.data[:, :10] = 0
    model.train()
    model.loss(data[:, :-1], data[:, 1:], k=2)  # populate activation statistics
    for _ in range(3):
        tracker.mark_sleep(threshold=1e-9)
    plan = tracker.prune_plan(sleeps=3)
    assert plan is not None
    pruned = growth.apply(model, "prune_mlp", data, k=2, max_params=10**8, max_delta_loss=0.01, keep=plan)
    assert pruned.cfg.mlp_hidden == model.cfg.mlp_hidden - 10
    assert growth.behaviour_delta(model, pruned, data, k=2)[0] < 1e-5


def test_unroll_core_at_k1_computes_what_the_loop_did_at_k(trained_tiny):
    model, _, data = trained_tiny
    new, event = growth.unroll_core(model, times=3)
    assert new.cfg.n_core == 3 * model.cfg.n_core and event["params_after"] > event["params_before"]
    dlogits, dloss = growth.behaviour_delta(model, new, data, k=3, k_new=1)
    assert dlogits < 1e-4 and dloss <= 0.01
    assert new.num_params() == count_params(new.cfg)


def test_unroll_core_with_noise_starts_the_copies_apart(trained_tiny):
    model, _, data = trained_tiny
    new, event = growth.unroll_core(model, times=3, noise=0.01)
    n = model.cfg.n_core
    first, later = new.core[0].attn.q.weight, new.core[n].attn.q.weight
    assert torch.equal(first, model.core[0].attn.q.weight) and not torch.equal(first, later)
    assert growth.behaviour_delta(model, new, data, k=3, k_new=1)[1] <= 0.01
    assert event["noise"] == 0.01


def test_renew_attention_silences_the_block_and_gives_it_fresh_readers(trained_tiny):
    model, _, data = trained_tiny
    new, event = growth.renew_attention(model, where=[0])
    attn = new.core[0].attn
    assert attn.o.weight.abs().sum() == 0 and not torch.equal(attn.q.weight, model.core[0].attn.q.weight)
    assert torch.equal(new.prelude[0].attn.q.weight, model.prelude[0].attn.q.weight)
    assert event["where"] == ["core.0"] and new.num_params() == model.num_params()
