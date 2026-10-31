import pytest
import torch

from cerbeto.model.student import Student, StudentConfig

TINY = StudentConfig(vocab_size=512, d_model=64, n_heads=4, head_dim=16, mlp_hidden=128,
                     n_prelude=1, n_core=1, n_coda=1, max_seq_len=64)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the GPU / takes a while")


@pytest.fixture
def tiny_cfg():
    return TINY


@pytest.fixture
def trained_tiny():
    """A tiny student after a few optimizer steps, so activations and moments are non-trivial."""
    torch.manual_seed(0)
    model = Student(TINY, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = torch.randint(0, TINY.vocab_size, (8, 33))
    for _ in range(5):
        opt.zero_grad()
        model.loss(data[:, :-1], data[:, 1:], k=2).backward()
        opt.step()
    return model, opt, data
