"""Spec 0021: a loop inside an unrolled core (`n_loop`): only the last core blocks run K times."""
import torch

from cerbeto.model import growth
from cerbeto.model.student import Student


def looped(model: Student, n: int) -> Student:
    """`model` with n blocks of zero output added at the end of its core, and those n made a loop."""
    for j in range(n):
        model, _ = growth.grow_depth(model, like=j % len(model.core))
    model.cfg = model.cfg.with_(n_loop=n)
    return model


def test_a_loop_of_silent_blocks_answers_as_before_at_any_k(trained_tiny):
    model, _, data = trained_tiny
    new = looped(model, 2)
    for k in (1, 2, 3):
        dlogits, dloss = growth.behaviour_delta(model, new, data, k=1, k_new=k)
        assert dlogits < 1e-5 and dloss < 1e-6


def test_only_the_last_blocks_repeat(trained_tiny):
    model, _, data = trained_tiny
    new = looped(model, 2)
    with torch.no_grad():
        for b in new.core[-2:]:
            b.mlp.down.weight.normal_(0, 0.02)
    calls = []
    hooks = [b.register_forward_hook(lambda m, i, o, j=j: calls.append(j)) for j, b in enumerate(new.core)]
    new(data[:, :-1], 3)
    for h in hooks:
        h.remove()
    assert calls == [0] + [1, 2] * 3
    assert not torch.allclose(new(data[:, :-1], 1), new(data[:, :-1], 2))


def test_the_loop_written_out_unrolled_computes_the_same(trained_tiny):
    """What scripts/hf/to_llama.py does: prelude, the blocks run once, the loop K times, coda — as plain blocks."""
    model, _, data = trained_tiny
    new = looped(model, 2)
    with torch.no_grad():
        for b in new.core[-2:]:
            b.attn.o.weight.normal_(0, 0.02)
    flat = Student(new.cfg.with_(n_loop=0, n_core=1 + 2 * 3)).eval()
    flat.load_state_dict(new.state_dict(), strict=False)
    flat.core = torch.nn.ModuleList([new.core[0], *list(new.core[1:]) * 3])
    new.eval()
    assert torch.allclose(new(data[:, :-1], 3), flat(data[:, :-1], 1), atol=1e-6)
